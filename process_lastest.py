#!/usr/bin/env python3
# -*-coding:utf-8-*-

import os
import glob
import re
import json
import numpy as np
from datetime import datetime
from scipy.signal import butter, filtfilt, savgol_filter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csiread

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)

def process_single_task(args):
    """
    独立的工作进程函数，处理单个受试者的单对 (pcap, json) 文件
    """
    pcap_path, json_path = args
    results_csi = []
    results_hr = []
    
    try:
        # 1. 读取心率标签
        with open(json_path, 'r') as f:
            hr_data = json.load(f)
        hr_vals = hr_data.get("heart_rate", [])
        t_strs = hr_data.get("start_time", [])
        if not hr_vals: return [], []
        
        # 转换为时间戳数组，方便向量化计算
        hr_ts = np.array([datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() for t in t_strs])
        
        # 2. 读取 CSI 数据
        csi_obj = csiread.Nexmon(pcap_path, chip='43455c0', bw=80)
        csi_obj.read()
        if csi_obj.csi.shape[0] < 100: return [], []
        
        ts_array = csi_obj.sec + csi_obj.usec * 1e-6
        
        # 提取有效子载波 (192个)
        center = 128
        raw_amp = np.abs(np.concatenate((
            csi_obj.csi[:, center-96-4 : center-4],
            csi_obj.csi[:, center+4 : center+96+4]
        ), axis=1))
        
        fs = 20  # 采样率

        # 3. 全局信号处理 (解决边缘效应)
        filtered_amp = raw_amp - np.mean(raw_amp, axis=0)
        filtered_amp = butter_bandpass_filter(filtered_amp, 0.8, 2.17, fs, order=3)
        filtered_amp = savgol_filter(filtered_amp, window_length=15, polyorder=3, axis=0)
        
        # --- 新增：智能时区与系统时间修正 ---
        tz_diff_hours = round((ts_array[0] - hr_ts[0]) / 3600)
        tz_offset_seconds = tz_diff_hours * 3600 
        hr_ts_aligned = hr_ts + tz_offset_seconds
        
        # 4. 时间对齐与特征提取
        for k, target_t in enumerate(hr_ts_aligned):
            idx = np.argmin(np.abs(ts_array - target_t))
            time_diff = np.abs(ts_array[idx] - target_t)
            
            if time_diff > 5.0:
                continue
                
            if idx >= 100:
                # 截取前 100 帧作为一个完整的时间窗特征矩阵 (100, 192)
                block = filtered_amp[idx - 100 : idx, :]
                
                # 【修改点】直接保留矩阵和数值，不展平、不变字符串
                results_csi.append(block)
                results_hr.append(hr_vals[k])
                
    except Exception as e:
        print(f"[错误] 处理文件 {os.path.basename(pcap_path)} 时出错: {e}")
        
    return results_csi, results_hr

def main():
    csi_dir = r"E:\esp32c5\xinlv\Data_DS1_raspberry-main\Data"
    hr_dir = r"E:\esp32c5\xinlv\Data_DS1_smartwatch-main\Data"
    # 【修改点】后缀改为 .npy
    output_csi = "training_data.npy"
    output_hr = "hr_data.npy"

    hr_subs = set(os.listdir(hr_dir))
    csi_subs = set(os.listdir(csi_dir))
    common_subs = sorted(list(hr_subs.intersection(csi_subs)))
    
    print(f"找到 {len(common_subs)} 个匹配的测试者，开始生成任务流...")
    
    tasks = []
    for subject_id in common_subs:
        hr_path = os.path.join(hr_dir, subject_id)
        csi_path = os.path.join(csi_dir, subject_id)
        json_files = glob.glob(os.path.join(hr_path, "*.json"))
        
        for json_f in json_files:
            match = re.search(r"(\d{4}_\d{2}_\d{2}_-_\d{2}_\d{2}_\d{2})", os.path.basename(json_f))
            if not match: continue
            match_key = match.group(1)
            
            pcap_candidates = glob.glob(os.path.join(csi_path, f"*{match_key}*.pcap"))
            if pcap_candidates:
                tasks.append((pcap_candidates[0], json_f))

    total_tasks = len(tasks)
    print(f"共生成 {total_tasks} 个处理任务，启动多进程池...")

    total_samples = 0
    all_csi = []
    all_hr = []
    
    # 启动进程池
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_task = {executor.submit(process_single_task, task): task for task in tasks}
        
        # 【修改点】不再逐行写文件，而是先收集到列表中
        for i, future in enumerate(as_completed(future_to_task), 1):
            try:
                csi_blocks, hr_vals = future.result()
                if csi_blocks:
                    all_csi.extend(csi_blocks)
                    all_hr.extend(hr_vals)
                    total_samples += len(csi_blocks)
            except Exception as e:
                print(f"提取任务结果时发生异常: {e}")
                
            # 打印进度条
            if i % 10 == 0 or i == total_tasks:
                print(f"进度: {i}/{total_tasks} 个文件处理完毕. 当前累计有效样本: {total_samples}")

    print(f"\n全部提取完成！共 {total_samples} 个样本。正在保存为 numpy 二进制文件...")
    # 【修改点】统一保存为 npy 格式
    np.save(output_csi, np.array(all_csi, dtype=np.float32))
    np.save(output_hr, np.array(all_hr, dtype=np.float32))
    print("保存成功！")

if __name__ == '__main__':
    main()