#!/usr/bin/env python3
# -*-coding:utf-8-*-

import sys
import json
import argparse
import numpy as np
import os
import glob
from datetime import datetime
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.interpolate import interp1d

try:
    import csiread
except ImportError:
    print("Error: csiread module not found. Please install it.")
    csiread = None

# ==============================================================================
# 1. 信号处理函数
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
# 2. 核心处理逻辑 (修复时间轴版)
# ==============================================================================
def process_aligned_dataset(csi_root, hr_root, output_csi="training_data1.txt", output_hr="hr_data1.txt"):
    
    # 清空并初始化输出文件
    open(output_csi, 'w').close()
    open(output_hr, 'w').close()

    f_csi = open(output_csi, 'a')
    f_hr = open(output_hr, 'a')
    
    total_samples = 0
    total_files_matched = 0

    if not os.path.exists(hr_root):
        print(f"错误: 手表数据路径不存在: {hr_root}")
        return

    subject_dirs = [d for d in os.listdir(hr_root) if os.path.isdir(os.path.join(hr_root, d))]
    print(f"找到受试者列表: {subject_dirs}")

    for subject_id in subject_dirs:
        hr_subject_path = os.path.join(hr_root, subject_id)
        csi_subject_path = os.path.join(csi_root, subject_id)

        if not os.path.exists(csi_subject_path):
            continue

        json_files = glob.glob(os.path.join(hr_subject_path, "*.json"))
        
        for json_path in json_files:
            try:
                # 解析文件名
                json_filename = os.path.basename(json_path)
                parts = json_filename.split('_HeartRateData')[0] 
                timestamp_sig = parts.split('_', 1)[1] 
                
                # 查找对应的 CSI 文件
                pcap_search_pattern = os.path.join(csi_subject_path, f"*{timestamp_sig}*.pcap")
                pcap_candidates = glob.glob(pcap_search_pattern)

                if not pcap_candidates:
                    print(f"[{subject_id}] 未找到对应的 CSI 文件: {timestamp_sig}")
                    continue
                
                pcap_path = pcap_candidates[0]
                print(f"处理: [{subject_id}] {os.path.basename(pcap_path)}")

                # === 读取心率 ===
                with open(json_path, 'r') as jf:
                    hr_data_content = json.load(jf)
                
                hr_values = hr_data_content.get("heart_rate", [])
                time_strs = hr_data_content.get("start_time", [])

                if not hr_values or not time_strs or len(hr_values) != len(time_strs):
                    continue

                # 构建时间轴 (心率)
                hr_timestamps = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() for t in time_strs]
                hr_func = interp1d(hr_timestamps, hr_values, kind='linear', bounds_error=False, fill_value=np.nan)
                
                # 获取本次采集的“基准绝对时间” (以心率记录的开始时间为准)
                base_start_time = hr_timestamps[0]
                hr_max_time = hr_timestamps[-1]

                # === 读取 CSI ===
                csi_reader = csiread.Nexmon(pcap_path, chip='43455c0', bw=80)
                csi_reader.read()
                raw_csi = csi_reader.csi
                
                if raw_csi.shape[0] == 0: continue

                # === 关键修正：计算 CSI 的相对时间 ===
                # 获取原始时间戳
                raw_ts = csi_reader.sec + csi_reader.usec * 1e-6
                
                # 判断 CSI 时间戳是否正常 (如果全是0，或者非常小，说明是相对时间)
                # 策略：不管 CSI 原来是 UTC 还是开机时间，我们强制把它的第一帧
                # 对齐到心率的第一帧。
                
                if len(raw_ts) > 1 and (raw_ts[-1] - raw_ts[0]) > 0.0001:
                    # 如果 CSI 自带时间戳有变化，计算相对偏移量
                    relative_time = raw_ts - raw_ts[0]
                else:
                    # 如果时间戳坏了 (全是0)，强制使用 50Hz (0.02s) 推算
                    print("  ⚠️警告: 检测到无效时间戳，使用 50Hz 强制推算")
                    relative_time = np.arange(raw_csi.shape[0]) * 0.02

                # === 裁剪子载波 (256 -> 192) ===
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
                samples_count = 0
                
                for i in range(amplitudes.shape[0]):
                    # === 核心修正 ===
                    # 当前帧的真实时间 = 心率开始时间 + CSI的相对流逝时间
                    curr_real_time = base_start_time + relative_time[i]

                    # 检查是否超出了心率数据的覆盖范围
                    if curr_real_time > hr_max_time:
                        break # 后面的数据没有标签了，停止
                    
                    # 获取标签
                    est_hr = float(hr_func(curr_real_time))
                    if np.isnan(est_hr):
                        continue

                    processed_signal = savitzky_golay_smooth(
                        butter_bandpass_filter(
                            remove_dc(amplitudes[i], 50.0), 
                        0.8, 4.0, 50.0), 
                    15, 3)

                    f_csi.write(','.join(map(str, processed_signal)) + '\n')
                    f_hr.write(f"{est_hr:.2f}\n")
                    
                    samples_count += 1
                
                print(f"  -> 成功写入: {samples_count} 帧")
                if samples_count > 0:
                    total_samples += samples_count
                    total_files_matched += 1

            except Exception as e:
                print(f"Err: {e}")
                continue

    f_csi.close()
    f_hr.close()
    
    print(f"\n处理完成. 匹配文件: {total_files_matched}, 总样本数: {total_samples}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 你的默认路径
    parser.add_argument('--csi-dir', dest='csi_dir', default=r"D:\esp32c5\xinlv\Data_DS1_raspberry-main\Data_DS1_raspberry-main\Data")
    parser.add_argument('--hr-dir', dest='hr_dir', default=r"D:\esp32c5\xinlv\Data_DS1_smartwatch-main\Data_DS1_smartwatch-main\Data")

    args = parser.parse_args()
    process_aligned_dataset(args.csi_dir, args.hr_dir)