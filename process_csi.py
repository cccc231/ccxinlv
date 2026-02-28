import sys
import csv
import json
import argparse
import numpy as np
import serial
from io import StringIO
import ast
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.interpolate import interp1d
import os
import glob
from datetime import datetime
import csiread

def parse_csi_amplitudes(csi_str):
    #   提取CSI幅度
    # Convert the string into a Python list of ints
    values = ast.literal_eval(csi_str)  #将字符串转换为python列表

    # Reshape into (imag, real) pairs
    complex_pairs = np.array(values).reshape(-1, 2) #将列表转换为复数对

    # Convert to complex numbers: real + j*imag
    csi_complex = complex_pairs[:,1] + 1j * complex_pairs[:,0] #1:real, 0:imag

    # Compute amplitudes
    amplitudes = np.abs(csi_complex)    #np.abs()计算复数的模长
    return amplitudes


def remove_dc(signal, fs, lowcut=2.0, highcut=5.0, order=3):    #输入信号、采样频率、低截止频率、高截止频率、滤波器阶数
    """
    Remove DC and out-of-band noise using a 3rd-order Butterworth band-pass filter.
    Matches the method described in WiHear (2-5 Hz speaking band).
    
    Parameters
    ----------
    signal : array-like
        Input CSI amplitude or raw time-series signal.
    fs : float
        Sampling frequency (Hz).
    lowcut : float
        Low cutoff frequency (Hz), default = 2.0 Hz.
    highcut : float
        High cutoff frequency (Hz), default = 5.0 Hz.
    order : int
        Order of Butterworth filter, default = 3.
    
    Returns
    -------
    filtered : ndarray
        Band-pass filtered signal.
    """
    #  Butterworth band-pass filter 巴特沃斯带通滤波器
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    filtered = filtfilt(b, a, signal)
    return filtered


def butter_bandpass_filter(signal: np.ndarray,
                           lowcut: float,
                           highcut: float,
                           fs: float,
                           order: int = 3) -> np.ndarray:
    """
    Pulse Extraction: 3rd-order Butterworth bandpass (default order=3).
    Uses zero-phase filtering (filtfilt).
    lowcut/highcut in Hz. fs is sampling frequency (Hz).
    """
    x = np.asarray(signal, dtype=float).copy()
    if x.size == 0:
        return x
    nyq = 0.5 * fs
    if not (0 < lowcut < highcut < nyq):
        raise ValueError(f"Invalid bandpass: low={lowcut}, high={highcut}, Nyquist={nyq}")
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    # filtfilt for zero-phase
    return filtfilt(b, a, x)


def savitzky_golay_smooth(signal: np.ndarray,
                          window_length: int = 15,
                          polyorder: int = 3) -> np.ndarray:
    """
    signal: np.ndarray输入的原始信号数组(CSI幅度数据)
    window_length: int = 15,默认窗长为15
    polyorder: int = 3,阶数为3
    Pulse Shaping: Savitzky-Golay smoothing (preserve waveform shape).
    Ensures window_length is odd and less than signal length.
    If signal too short, returns original signal.
    """
    x = np.asarray(signal, dtype=float).copy()
    n = x.size
    if n == 0:
        return x
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    if wl < (polyorder + 2):
        raise ValueError("window_length too small for polyorder")
    if wl >= n:
        # fallback: choose the largest odd window smaller than n
        wl_candidate = n - 1
        if wl_candidate % 2 == 0:
            wl_candidate -= 1
        if wl_candidate < (polyorder + 2):
            # can't apply SG; return original
            return x
        wl = wl_candidate
    return savgol_filter(x, wl, polyorder)

def process_dataset(dataset_root, output_csi="training_data.txt", output_hr="hr_data.txt" ):
    
    print(f"开始处理数据集: {dataset_root}")
    print(f"输出文件: {output_csi}, {output_hr}")

    f_train = open(output_csi, 'w')
    f_hr = open(output_hr, 'w')

    total_samples = 0

    search_path = os.path.join(dataset_root, "Data_DS1_smartwatch-main","**","*.json")     #**表示当前目录下的所有子目录，*.json表示搜索所有json文件
    pcap_files = glob.glob(search_path, recursive=True)     #recursive=True表示递归搜索子目录

    if not pcap_files:
        print(f"错误: 未找到任何JSON文件在 {dataset_root}")
        return
        
    for pcap_path in pcap_files:
        dirname = os.path.dirname(pcap_path)    #os.path中的模块，dirname()返回路径的目录名
        filename = os.path.basename(pcap_path)      #basename()返回路径的文件名和后缀

        try:
            base_prefix = filename.split("_HeartRateData_")[0]
            timestamp_part = base_prefix[3:]    #去掉ID前缀，只保留时间

            json_candidates = glob.glob(os.path.join(dirname,f"*{timestamp_part}*HeartRateData.json"))
            
            if not json_candidates:
                timestamp_part = base_prefix.split("_")[1]
                json_candidates = glob.glob(os.path.join(dirname,f"*{timestamp_part}*HeartRateData.json"))

            json_path = json_candidates[0]
        
        except Exception as e:
            print(f"文件名解析错误 {filename}: {e}")
            continue


    