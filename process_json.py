#!/usr/bin/env python3
# -*-coding:utf-8-*-

import sys
import csv
import json
import argparse
import numpy as np
import os
import glob
from datetime import datetime
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.interpolate import interp1d

# 尝试导入 csiread
try:
    import csiread
except ImportError:
    csiread = None

# ==============================================================================
# 核心信号处理函数 (保持不变)
# ==============================================================================
def remove_dc(signal, fs, lowcut=2.0, highcut=5.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    filtered = filtfilt(b, a, signal)
    return filtered

def butter_bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 3) -> np.ndarray:
    x = np.asarray(signal, dtype=float).copy()
    if x.size == 0: return x
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)

def savitzky_golay_smooth(signal: np.ndarray, window_length: int = 15, polyorder: int = 3) -> np.ndarray:
    x = np.asarray(signal, dtype=float).copy()
    n = x.size
    if n == 0: return x
    wl = int(window_length)
    if wl % 2 == 0: wl += 1
    if wl >= n: wl = n - 1 if (n - 1) % 2 != 0 else n - 2
    if wl < (polyorder + 2): return x
    return savgol_filter(x, wl, polyorder)

# ==============================================================================
# 处理 e-health 数据集的函数 (支持双路径)
# ==============================================================================
def process_ehealth_dataset(csi_root, hr_root, output_csi="training_data.txt", output_hr="hr_data.txt"):
    if csiread is None:
        print("错误: 请先安装 csiread 库 (pip install csiread) 以处理 PCAP 文件。")
        return

    print(f"CSI(树莓派) 路径: {csi_root}")
    print(f"心率(手表) 路径: {hr_root}")
    
    # 清空输出文件
    open(output_csi, 'w').close()
    open(output_hr, 'w').close()

    f_csi = open(output_csi, 'a')
    f_hr = open(output_hr, 'a')
    
    total_samples = 0

    # 1. 查找所有 PCAP 文件
    search_path = os.path.join(csi_root, "**", "*.pcap")
    pcap_files = glob.glob(search_path, recursive=True)
    
    if not pcap_files:
        print("未找到 .pcap 文件，请检查 CSI 路径。")
        return
    
    print(f"找到 {len(pcap_files)} 个 PCAP 文件，开始匹配心率数据...")

    for pcap_path in pcap_files:
        filename = os.path.basename(pcap_path)
        
        # 解析文件名: 01_2022_05_31_-_18_22_20_bw_80_ch_36.pcap
        try:
            base_prefix = filename.split("_bw_")[0] # 得到 01_2022_05_31_-_18_22_20
            
            # 提取时间戳部分用于匹配: 2022_05_31_-_18_22_20
            # 假设前缀 ID 长度不固定，我们取第一个下划线后的部分作为特征
            parts = base_prefix.split('_', 1) 
            if len(parts) < 2: continue
            timestamp_signature = parts[1] # 2022_05_31_-_18_22_20
            
        except Exception as e:
            continue

        # 2. 在 hr_root 中查找对应的 JSON 文件
        # 搜索模式: *2022_05_31_-_18_22_20*HeartRateData.json
        # 这样无论 ID 是 01 还是 1 都能匹配
        json_pattern = os.path.join(hr_root, "**", f"*{timestamp_signature}*HeartRateData.json")
        json_candidates = glob.glob(json_pattern, recursive=True)
        
        if not json_candidates:
            # 没找到，跳过
            # print(f"未找到匹配的心率文件: {timestamp_signature}")
            continue
            
        json_path = json_candidates[0] # 取第一个匹配的

        # 3. 读取并处理数据
        try:
            # --- 读取心率 ---
            with open(json_path, 'r') as jf:
                hr_data = json.load(jf)
            hr_values = hr_data.get("heart_rate", [])
            time_strs = hr_data.get("start_time", [])
            
            if not hr_values or len(hr_values) != len(time_strs): continue
            
            hr_timestamps = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() for t in time_strs]
            hr_func = interp1d(hr_timestamps, hr_values, kind='linear', fill_value="extrapolate")

            # --- 读取 CSI ---
            csi_data = csiread.Nexmon(pcap_path, chip='43455c0', bw=80)
            csi_data.read()
            raw_csi = csi_data.csi
            if raw_csi.shape[0] == 0: continue

            # --- 维度截取 (256 -> 192) ---
            if raw_csi.shape[1] >= 192:
                start_idx = (raw_csi.shape[1] - 192) // 2
                csi_complex = raw_csi[:, start_idx : start_idx + 192]
            else:
                continue

            amplitudes = np.abs(csi_complex)
            raw_timestamps = csi_data.sec + csi_data.usec * 1e-6
            
            # 如果时间戳全是 0 或者异常，使用索引作为时间轴 (假设采样率 50Hz -> 0.02s 一帧)
            if np.all(raw_timestamps == 0) or len(raw_timestamps) < 2:
                # 这是一个兜底策略，假设 50Hz
                relative_time = np.arange(len(amplitudes)) * 0.02
            else:
                # 计算相对时间：当前时间 - 第一帧时间
                relative_time = raw_timestamps - raw_timestamps[0]

            # 2. 心率数据的基准时间
            # 我们假设 PCAP 的开始时间 就是 JSON 里的 start_time[0]
            # (这是一个很强的假设，但对于同一次采集的数据通常有效)
            base_time = hr_timestamps[0]

            # 3. 写入文件
            samples_written_this_file = 0
            
            for i in range(amplitudes.shape[0]):
                # 信号处理
                processed = savitzky_golay_smooth(
                    butter_bandpass_filter(
                        remove_dc(amplitudes[i], 20.0), 
                    0.8, 2.17, 20.0), 
                15, 3)
                
                # 计算当前帧对应的真实世界时间
                # 绝对时间 = 基准时间(来自JSON) + 相对时间偏移(来自CSI)
                current_real_time = base_time + relative_time[i]

                # 检查时间是否超出心率数据的覆盖范围
                if current_real_time > hr_timestamps[-1]:
                    break # 超出范围，停止
                
                # 插值获取心率
                try:
                    est_hr = float(hr_func(current_real_time))
                except ValueError:
                    continue 

                f_csi.write(','.join(map(str, processed)) + '\n')
                f_hr.write(f"{est_hr:.2f}\n")
                samples_written_this_file += 1
            
            # total_samples 只统计真正写入的
            total_samples += samples_written_this_file
            print(f"匹配成功: {filename} ({samples_written_this_file} 帧写入)")
            
            total_samples += amplitudes.shape[0]
            print(f"匹配成功: {filename} <-> {os.path.basename(json_path)} ({amplitudes.shape[0]} 帧)")

        except Exception as e:
            print(f"处理出错 {filename}: {e}")
            continue

    f_csi.close()
    f_hr.close()
    print(f"\n========================================")
    print(f"处理全部完成!")
    print(f"总共生成样本数: {total_samples}")
    print(f"生成文件: {output_csi}, {output_hr}")
    print(f"========================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 两个参数：一个指明 CSI 在哪，一个指明 心率在哪
    parser.add_argument('--csi-dir', dest='csi_dir', required=True,
                        help="树莓派PCAP数据文件夹路径")
    parser.add_argument('--hr-dir', dest='hr_dir', required=True,
                        help="手表JSON数据文件夹路径")

    args = parser.parse_args()
    
    process_ehealth_dataset(args.csi_dir, args.hr_dir)