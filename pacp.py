import csiread
import pandas as pd
import numpy as np
import os
import struct
from datetime import datetime

# --- 新增函数：手动从 PCAP 文件提取时间戳 ---
def get_pcap_timestamps(pcap_file_path):
    """
    不依赖 csiread，直接读取 PCAP 文件头获取每帧的时间戳
    """
    timestamps = []
    try:
        with open(pcap_file_path, 'rb') as f:
            # 1. 读取全局文件头 (24 bytes)
            global_header = f.read(24)
            if len(global_header) < 24:
                return None
            
            # 判断大小端模式 (Magic Number)
            magic_number = global_header[:4]
            if magic_number == b'\xa1\xb2\xc3\xd4':
                endian = '>' # Big Endian
            else:
                endian = '<' # Little Endian (常见)

            while True:
                # 2. 读取每个数据包的包头 (16 bytes)
                # struct pcap_pkthdr {
                #     bpf_u_int32 ts_sec;  /* timestamp seconds */
                #     bpf_u_int32 ts_usec; /* timestamp microseconds */
                #     bpf_u_int32 caplen;  /* length of portion present */
                #     bpf_u_int32 len;     /* length this packet (off wire) */
                # };
                header_data = f.read(16)
                if len(header_data) < 16:
                    break # 文件结束
                
                ts_sec, ts_usec, incl_len, orig_len = struct.unpack(endian + 'IIII', header_data)
                
                # 计算时间戳 (秒 + 微秒)
                current_ts = ts_sec + ts_usec / 1_000_000.0
                timestamps.append(current_ts)

                # 3. 跳过数据包内容，直接去下一个包头
                f.seek(incl_len, 1)
                
        return np.array(timestamps)
    except Exception as e:
        print(f"⚠️ 警告: 手动提取时间戳失败 - {e}")
        return None

# --- 主处理函数 ---
def pcap_to_csv_raw(pcap_file, output_csv, chip_model='43455c0', bandwidth=80):
    
    # 1. 检查文件
    if not os.path.exists(pcap_file):
        print(f"❌ 错误: 找不到文件 {pcap_file}")
        return

    print(f"正在读取文件: {os.path.basename(pcap_file)} ...")
    
    # 2. 使用 csiread 读取 CSI 数据 (复数)
    try:
        csidata = csiread.Nexmon(pcap_file, chip=chip_model, bw=bandwidth)
        csidata.read()
    except Exception as e:
        print(f"❌ csiread 读取错误: {e}")
        return

    if csidata.csi is None or len(csidata.csi) == 0:
        print("⚠️  警告: 未提取到 CSI 数据。")
        return

    csi_matrix = csidata.csi
    num_packets, num_subcarriers = csi_matrix.shape
    print(f"✅ CSI 提取成功: {num_packets} 个数据包")

    # 3. [修复部分] 手动读取时间戳
    print("正在提取时间戳...")
    timestamps = get_pcap_timestamps(pcap_file)
    
    # 校验时间戳数量是否与 CSI 包数量一致
    # 注意：如果文件里包含非 CSI 数据包，这里可能会有数量差异，通常 Nexmon 文件是纯净的
    time_column = []
    if timestamps is not None:
        if len(timestamps) == num_packets:
            print(f"✅ 时间戳对齐成功 ({len(timestamps)} 帧)")
            time_column = timestamps
        else:
            print(f"⚠️ 数据包数量不匹配 (CSI: {num_packets}, Time: {len(timestamps)})")
            print("   -> 将尝试截取或填充以匹配 CSI 数据")
            if len(timestamps) > num_packets:
                time_column = timestamps[:num_packets]
            else:
                # 如果时间戳少了，后面补 None
                time_column = list(timestamps) + [None] * (num_packets - len(timestamps))
    else:
        print("⚠️ 无法获取时间戳，将不包含时间列")

    # 4. 构建表格
    print("正在构建 CSV 表格...")
    column_names = [f'Sub_{i}' for i in range(num_subcarriers)]
    df = pd.DataFrame(csi_matrix, columns=column_names)

    # 插入索引
    df.insert(0, 'Packet_Index', range(num_packets))
    
    # 插入时间戳 (如果获取成功)
    if len(time_column) > 0:
        df.insert(1, 'Timestamp', time_column)
        # 可选：再加一列可读的时间字符串
        try:
            readable_time = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if ts else "" for ts in time_column]
            df.insert(2, 'Time_String', readable_time)
        except:
            pass

    # 5. 保存
    print(f"正在保存到 {output_csv} ...")
    df.to_csv(output_csv, index=False)
    print(f"🎉 处理完成！")

# --- 执行配置 ---
input_file = r'D:\esp32c5\xinlv\Data_DS1_raspberry-main\Data_DS1_raspberry-main\Data\062\4_2022_07_12_-_15_14_56_bw_80_ch_36.pcap'
output_file = 'csi_data_with_fixed_time062.csv'

pcap_to_csv_raw(input_file, output_file, chip_model='43455c0', bandwidth=80)