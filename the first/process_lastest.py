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
    返回处理后的特征字符串列表和标签字符串列表
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
        
        # 提取有效子载波 (184个)
        center = 128
        raw_amp = np.abs(np.concatenate((
            csi_obj.csi[:, center-96-4 : center-4],
            csi_obj.csi[:, center+4 : center+96+4]
        ), axis=1))
        
        fs = 20  # 采样率

        # 3. 全局信号处理 (解决边缘效应)
        # 3.1 去直流 (减去列均值)
        filtered_amp = raw_amp - np.mean(raw_amp, axis=0)
        # 3.2 提取心率频带 (0.8 - 2.17 Hz)
        filtered_amp = butter_bandpass_filter(filtered_amp, 0.8, 2.17, fs, order=3)
        # 3.3 平滑去噪
        filtered_amp = savgol_filter(filtered_amp, window_length=15, polyorder=3, axis=0)
        
        # --- 新增：智能时区与系统时间修正 ---
        # 自动计算两个设备之间相差的“小时数”（四舍五入）
        tz_diff_hours = round((ts_array[0] - hr_ts[0]) / 3600)
        # 将小时数转换回秒，这就是我们需要补偿的宏观时间差
        tz_offset_seconds = tz_diff_hours * 3600 
        # 对心率时间戳进行宏观对齐
        hr_ts_aligned = hr_ts + tz_offset_seconds
        
        # 4. 时间对齐与特征提取
        for k, target_t in enumerate(hr_ts_aligned):
            idx = np.argmin(np.abs(ts_array - target_t))
            time_diff = np.abs(ts_array[idx] - target_t)
            
            # 【关键修改】容差放宽到 5.0 秒。
            # 因为即使补齐了时区，实验人员分别启动两个设备也会有 1~4 秒的手速延迟
            if time_diff > 5.0:
                continue
                
            if idx >= 100:
                # 截取前 100 帧作为一个完整的时间窗特征矩阵 (100, 184)
                block = filtered_amp[idx - 100 : idx, :]
                
                # 将矩阵展平为一维向量，保留时序与空间的整体信息
                block_flat = block.flatten()
                
                # 格式化并保存到内存列表
                csi_str = ",".join([f"{val:.4f}" for val in block_flat])
                hr_str = f"{hr_vals[k]:.2f}"
                
                results_csi.append(csi_str)
                results_hr.append(hr_str)
                
    except Exception as e:
        # 捕捉异常并打印，防止静默失败
        print(f"[错误] 处理文件 {os.path.basename(pcap_path)} 时出错: {e}")
        
    return results_csi, results_hr

def main():
    csi_dir = r"D:\esp32c5\xinlv\Data_DS1_raspberry-main\Data_DS1_raspberry-main\Data"
    hr_dir = r"D:\esp32c5\xinlv\Data_DS1_smartwatch-main\Data_DS1_smartwatch-main\Data"
    output_csi = "training_data5.txt"
    output_hr = "hr_data5.txt"
    
    # 清空输出文件
    for f in [output_csi, output_hr]:
        with open(f, 'w') as _: pass

    hr_subs = set(os.listdir(hr_dir))
    csi_subs = set(os.listdir(csi_dir))
    common_subs = sorted(list(hr_subs.intersection(csi_subs)))
    
    print(f"找到 {len(common_subs)} 个匹配的测试者，开始生成任务流...")
    
    # 收集所有需要处理的任务 (pcap_path, json_path)
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
    # 启动进程池 (默认使用 CPU 核心数)
    with ProcessPoolExecutor(max_workers=4) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_task, task): task for task in tasks}
        
        # 主进程负责统一写入文件，避免多进程抢占写入导致数据错乱
        with open(output_csi, 'a') as f_csi, open(output_hr, 'a') as f_hr:
            for i, future in enumerate(as_completed(future_to_task), 1):
                try:
                    csi_lines, hr_lines = future.result()
                    if csi_lines:
                        f_csi.write("\n".join(csi_lines) + "\n")
                        f_hr.write("\n".join(hr_lines) + "\n")
                        total_samples += len(csi_lines)
                except Exception as e:
                    print(f"提取任务结果时发生异常: {e}")
                    
                # 打印进度条
                if i % 10 == 0 or i == total_tasks:
                    print(f"进度: {i}/{total_tasks} 个文件处理完毕. 当前累计有效样本: {total_samples}")

    print(f"\n全部完成！共提取并保存了 {total_samples} 个有效特征-标签对。")

if __name__ == '__main__':
    # 为了防止 Windows 下多进程报错，需要包裹在 __main__ 下
    main()