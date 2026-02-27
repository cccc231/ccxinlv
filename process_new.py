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

# 尝试导入 csiread
try:
    import csiread
except ImportError:
    print("Error: csiread module not found. Please install it.")
    csiread = None

# ==============================================================================
# 1. 信号处理函数
# ==============================================================================
def remove_dc(signal, fs, lowcut=0.8, highcut=4, order=3):
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
# 2. 核心处理逻辑
# ==============================================================================
def process_data(csi_root, hr_root, output_csi="training_data5.txt", output_hr="hr_data5.txt"):
    
    # 初始化清空文件
    open(output_csi, 'w').close()
    open(output_hr, 'w').close()

    f_csi = open(output_csi, 'a')
    f_hr = open(output_hr, 'a')
    
    total_samples = 0
    total_files_matched = 0

    # ---------------------------------------------------------
    # 主循环:遍历手表数据目录
    # ---------------------------------------------------------
    if not os.path.exists(hr_root):
        print(f"错误: 手表数据路径不存在: {hr_root}")
        return

    subject_dirs = [d for d in os.listdir(hr_root) if os.path.isdir(os.path.join(hr_root, d))]
    
    for subject_id in subject_dirs:
        hr_subject_path = os.path.join(hr_root, subject_id)
        csi_subject_path = os.path.join(csi_root, subject_id)

        # 如果树莓派数据里没有这个人的文件夹，跳过
        if not os.path.exists(csi_subject_path):
            continue 

        json_files = glob.glob(os.path.join(hr_subject_path, "*.json"))
        
        for json_path in json_files:
            try:
                # 从 JSON 文件名提取唯一时间标识 (例如: 2022_06_29_-_11_35_52)
                json_filename = os.path.basename(json_path)
                parts = json_filename.split('_HeartRateData')[0] 
                match_key = parts.split('_', 1)[1] 
                
                # 去树莓派目录找包含这个时间标识的 .pcap 文件
                pcap_search_pattern = os.path.join(csi_subject_path, f"*{match_key}*.pcap")
                pcap_candidates = glob.glob(pcap_search_pattern)

                if not pcap_candidates:
                    continue
                
                pcap_path = pcap_candidates[0]
                pcap_filename = os.path.basename(pcap_path)
                print(f"正在处理: [{subject_id}] {pcap_filename}")

                # 直接使用 match_key (即 2022_06_29_-_11_35_52) 解析时间
                try:
                    anchor_dt = datetime.strptime(match_key, "%Y_%m_%d_-_%H_%M_%S")
                    anchor_timestamp = anchor_dt.timestamp()
                except ValueError:
                    print(f"  -> 文件名时间格式解析失败: {match_key}，跳过")
                    continue

                # 读取心率数据
                with open(json_path, 'r') as jf:
                    hr_data_content = json.load(jf)
                hr_values = hr_data_content.get("heart_rate", [])
                time_strs = hr_data_content.get("start_time", [])

                if not hr_values or not time_strs: continue

                # 构建心率插值函数
                hr_timestamps = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() for t in time_strs]
                hr_func = interp1d(hr_timestamps, hr_values, kind='linear', bounds_error=False, fill_value=np.nan)
                
                hr_start = hr_timestamps[0]
                hr_end = hr_timestamps[-1]

                # 读取 CSI 数据
                csi_reader = csiread.Nexmon(pcap_path, chip='43455c0', bw=80)
                csi_reader.read()
                raw_csi = csi_reader.csi
                if raw_csi.shape[0] == 0: continue
                # 获取pcap文件原始时间戳
                raw_timestamps = csi_reader.sec + csi_reader.usec * 1e-6

                fs = 8.4
                
                # 锚点时间 + (当前帧原始时间 - 第一帧原始时间)
                corrected_timestamps = anchor_timestamp + (raw_timestamps - raw_timestamps[0])

                # 诊断输出
                print(f"  [修正] CSI数据起点: {datetime.fromtimestamp(corrected_timestamps[0])}")
                print(f"  [参考] HR数据起点  : {datetime.fromtimestamp(hr_start)}")

                # 裁剪子载波 (256 -> 192)
                if raw_csi.shape[1] >= 250:
                    center_idx = 128
                    dc_gap = 4
                    csi_complex = np.concatenate((
                        raw_csi[:, center_idx - 96 - dc_gap : center_idx - dc_gap],
                        raw_csi[:, center_idx + dc_gap : center_idx + 96 + dc_gap]
                    ), axis=1)
                else: continue
                
                amplitudes = np.abs(csi_complex)
                samples_written = 0

                # ---------------------------------------------------------
                # 写入时的交集过滤
                # ---------------------------------------------------------
                for i in range(amplitudes.shape[0]):
                    curr_ts = corrected_timestamps[i]

                    # 严格只保留重叠部分 (Intersection)
                    if curr_ts < hr_start: continue # 还没到心率记录开始时间
                    if curr_ts > hr_end: break      # 心率记录结束了
                    
                    est_hr = float(hr_func(curr_ts))
                    if np.isnan(est_hr): continue

                    nyquist = 0.5 * fs

                    # 信号处理 (滤波)
                    processed = savitzky_golay_smooth(
                        butter_bandpass_filter(remove_dc(amplitudes[i], fs), 0.8, 2.17, fs), 
                    15, 3)

                    f_csi.write(','.join(map(str, processed)) + '\n')
                    f_hr.write(f"{est_hr:.2f}\n")
                    samples_written += 1
                
                print(f"  -> 成功写入: {samples_written} 帧")
                if samples_written > 0:
                    total_samples += samples_written
                    total_files_matched += 1

            except Exception as e:
                print(f"Err: {e}")
                continue

    f_csi.close()
    f_hr.close()
    print(f"\n全部处理完成! 总匹配文件: {total_files_matched}, 总样本数: {total_samples}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 请确认这里的路径是你电脑上的实际路径
    parser.add_argument('--csi-dir', dest='csi_dir', default=r"D:\esp32c5\xinlv\Data_DS1_raspberry-main\Data_DS1_raspberry-main\Data")
    parser.add_argument('--hr-dir', dest='hr_dir', default=r"D:\esp32c5\xinlv\Data_DS1_smartwatch-main\Data_DS1_smartwatch-main\Data")
    
    args = parser.parse_args()
    process_data(args.csi_dir, args.hr_dir)