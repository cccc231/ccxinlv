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
# 核心信号处理函数
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
# 处理 e-health 数据集的函数
# ==============================================================================
def process_ehealth_dataset(csi_root, hr_root, output_csi="training_data.txt", output_hr="hr_data.txt"):
    
    # 清空输出文件
    open(output_csi, 'w').close()
    open(output_hr, 'w').close()

    f_csi = open(output_csi, 'a')
    f_hr = open(output_hr, 'a')
    
    total_samples = 0

    # 1. 查找所有 PCAP 文件
    search_path = os.path.join(csi_root, "**", "*.pcap")
    pcap_files = glob.glob(search_path, recursive=True)

    for pcap_path in pcap_files:
        filename = os.path.basename(pcap_path)
        
        # 解析文件名
        try:
            base_prefix = filename.split("_bw_")[0]
            parts = base_prefix.split('_', 1) 
            if len(parts) < 2: continue
            timestamp_signature = parts[1]
            
        except Exception as e:
            continue

        # 2. 在 hr_root 中查找对应的 JSON 文件
        json_pattern = os.path.join(hr_root, "**", f"*{timestamp_signature}*HeartRateData.json")
        json_candidates = glob.glob(json_pattern, recursive=True)
        
        if not json_candidates:
            continue
            
        json_path = json_candidates[0]

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
            if raw_csi.shape[1] >= 250:
                center_idx = 128
                dc_gap = 4
                idx_left_start = center_idx - 96 - dc_gap
                idx_left_end = center_idx - dc_gap
                csi_left = raw_csi[:, idx_left_start : idx_left_end]
                
                idx_right_start = center_idx + dc_gap
                idx_right_end = center_idx + 96 + dc_gap
                csi_right = raw_csi[:, idx_right_start : idx_right_end]
                csi_complex = np.concatenate((csi_left, csi_right), axis=1)
            else:
                continue

            amplitudes = np.abs(csi_complex)
            raw_timestamps = csi_data.sec + csi_data.usec * 1e-6    #从pcap中提取时间戳
            
            # 计算时间间隔：当前时间 - 第一帧时间
            relative_time = raw_timestamps - raw_timestamps[0]

            # 2. 心率数据的基准时间，开始时间是 JSON 里的 start_time[0]
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
                
                # 计算当前pcap文件中每帧所对应的时间
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
            
            total_samples += samples_written_this_file
            print(f"匹配成功: {filename} <-> {os.path.basename(json_path)} (写入: {samples_written_this_file}/{amplitudes.shape[0]} 帧)")

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
    
    parser.add_argument('--csi-dir', dest='csi_dir', required=True,
                        help="树莓派PCAP数据文件夹路径")
    parser.add_argument('--hr-dir', dest='hr_dir', required=True,
                        help="手表JSON数据文件夹路径")

    args = parser.parse_args()
    
    process_ehealth_dataset(args.csi_dir, args.hr_dir)