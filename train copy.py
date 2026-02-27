import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import datetime
import pandas as pd

# 只有在 Windows 上才会有 dll 加载问题
if os.name == 'nt':
    try:
        # 尝试手动加载一下 CUDA 库，看看报错是什么
        ctypes = __import__('ctypes')
        # 这里的名字是 TF 2.10 需要的特定版本
        ctypes.windll.LoadLibrary('cudart64_110.dll')
        print("✅ 恭喜：CUDA 动态库加载成功！")
    except OSError:
        print("❌ 失败：找不到 'cudart64_110.dll'。")
        print("   原因：你没有安装 CUDA 11.2 或没有配置环境变量。")
        print("   结论：TensorFlow 将自动切换回 CPU 模式。")

# ==========================================
# 1. 核心数据生成器 (支持随机打乱窗口)
# ==========================================
class CSI_Sequence(keras.utils.Sequence):
    def __init__(self, x_data, y_data, indices, batch_size, window_size):
        self.x_data = x_data  # 原始数据引用
        self.y_data = y_data  # 原始标签引用
        self.indices = indices  # 这一组乱序后的索引
        self.batch_size = batch_size
        self.window_size = window_size

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_inds = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for i in batch_inds:
            batch_x.append(self.x_data[i: i + self.window_size])
            batch_y.append(self.y_data[i])
        return np.array(batch_x), np.array(batch_y)

# --- 显卡配置 ---
gpus = tf.config.list_physical_devices('GPU')
print("当前检测到的 GPU:", gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- 全局配置 ---
MSE_THRESHOLD = 0.01
CONTINUE_TRAINING_MODEL = ""
BATCH_SIZE = 1024
WINDOW_SIZE = 100

# ==========================================
# 2. 读取数据 & 预计算标签
# ==========================================
print(f"Reading data with Pandas ({datetime.datetime.now()})...")

try:
    df = pd.read_csv("training_data5.txt", header=None)
    data_matrix = df.values.astype('float32') # Shape: (N, 192)

    df_hr = pd.read_csv("hr_data5.txt", header=None)
    series_hr = df_hr[0].astype('float32')

    print(f"Done reading. Data Shape: {data_matrix.shape}")

except FileNotFoundError:
    print("错误: 找不到 training_data5.txt 或 hr_data5.txt")
    exit()

print("正在计算滑动窗口平均值标签...")
y_all_avg = series_hr.rolling(window=WINDOW_SIZE).mean().shift(-(WINDOW_SIZE-1))
y_all_avg = y_all_avg.fillna(0).values

# ✅ 把方差打印放在这里，训练前就能看到
print(f"标签数据的方差 (基准线): {np.var(y_all_avg):.4f}")

# ==========================================
# 3. 数据划分 (基于索引随机打乱)
# ==========================================
print("正在进行基于索引的随机划分...")

valid_indices = np.arange(len(data_matrix) - WINDOW_SIZE)

# 🚨 关键：打乱索引
np.random.seed(42) 
np.random.shuffle(valid_indices)

split_idx = int(len(valid_indices) * 0.8)
train_indices = valid_indices[:split_idx]
val_indices = valid_indices[split_idx:]

print(f"训练集窗口数: {len(train_indices)}, 验证集窗口数: {len(val_indices)}")

train_ds = CSI_Sequence(data_matrix, y_all_avg, train_indices, BATCH_SIZE, WINDOW_SIZE)
val_ds = CSI_Sequence(data_matrix, y_all_avg, val_indices, BATCH_SIZE, WINDOW_SIZE)

# ==========================================
# 4. 模型构建 (无 BN 版本)
# ==========================================
class stopCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_loss') <= MSE_THRESHOLD) or os.path.isfile("training.stop"):
            print("\nReached {0} MSE; stopping training.".format(MSE_THRESHOLD))
            self.model.stop_training = True

if CONTINUE_TRAINING_MODEL != "":
    print(f"Loading saved model: {CONTINUE_TRAINING_MODEL}")
    model = keras.models.load_model(CONTINUE_TRAINING_MODEL)
else:
    main_input = keras.Input(shape=(WINDOW_SIZE, 192), name='main_input')

    # 回归原始结构：LSTM -> Dropout (无 BatchNormalization)
    layers = keras.layers.LSTM(64, return_sequences=True, name='lstm_1')(main_input)
    layers = keras.layers.Dropout(0.2, name='dropout_1')(layers)

    layers = keras.layers.LSTM(32, name='lstm_2')(layers)
    layers = keras.layers.Dropout(0.2, name='dropout_2')(layers)

    layers = keras.layers.Dense(16, activation='relu', name='dense_1')(layers)
    hr_output = keras.layers.Dense(1, name='hr_output')(layers)

    model = keras.Model(inputs=main_input, outputs=hr_output)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

model.summary()

# ==========================================
# 5. 开始训练 (带 Ctrl+C 保护)
# ==========================================
print(f"开始训练 (Configuration: Shuffle YES, Norm NO)...")
callbacks_list = [stopCallback()]

tf.keras.backend.clear_session()

try:
    model.fit(train_ds,
              epochs=100, # 这里设置你想要的轮数
              verbose=1,
              validation_data=val_ds,
              callbacks=callbacks_list)
except KeyboardInterrupt:
    print("\n\n🛑 训练被用户手动停止 (Ctrl+C)。正在保存模型并运行测试...")

# 无论是否跑完，都会保存模型
model.save("csi_hr5.keras")
print("Model saved to csi_hr5.keras")

# ==========================================
# 6. 验证集抽查 (Debug) - 修正版
# ==========================================
print("\n" + "=" * 40)
print("🔍 验证集预测抽查 (Debug Check)")
print("=" * 40)

try:
    # 修正：直接通过索引 [0] 获取验证集的第一个 Batch
    x_batch, y_batch = val_ds[0]

    print("正在进行推理预测...")
    preds = model.predict(x_batch, verbose=0)
    y_true = y_batch.flatten()  # 这里的 y_batch 已经是 numpy 数组了，不需要 .numpy()
    y_pred = preds.flatten()

    print(f"\n{'索引':<5} | {'真实呼吸率':<15} | {'模型预测':<15} | {'误差':<10}")
    print("-" * 55)

    for i in range(15):
        diff = abs(y_true[i] - y_pred[i])
        print(f"{i:<5} | {y_true[i]:<15.2f} | {y_pred[i]:<15.2f} | {diff:<10.2f}")

    print("-" * 55)
    print(f"预测值标准差 (Std): {np.std(y_pred):.4f}")

    if np.std(y_pred) > 0.5:
        print("✅ 模型状态良好，有波动能力。")
    else:
        print("⚠️ 警告：模型可能在输出死值。")

except Exception as e:
    print(f"调试代码运行时出错: {e}")