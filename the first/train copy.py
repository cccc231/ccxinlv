import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import datetime
import pandas as pd

# 只有在 Windows 上才会有 dll 加载问题
if os.name == 'nt':
    try:
        ctypes = __import__('ctypes')
        ctypes.windll.LoadLibrary('cudart64_110.dll')
        print("✅ 恭喜：CUDA 动态库加载成功！")
    except OSError:
        print("❌ 失败：找不到 'cudart64_110.dll'。")

# --- 显卡配置 ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- 全局配置 ---
MSE_THRESHOLD = 0.01
CONTINUE_TRAINING_MODEL = ""
BATCH_SIZE = 64
WINDOW_SIZE = 100

# ==========================================
# 1. 读取数据 & 结构化重塑
# ==========================================
print(f"Reading data ({datetime.datetime.now()})...")

try:
    # 加载 CSI 数据
    df = pd.read_csv("training_data5.txt", header=None)
    data_matrix = df.values.astype('float32') # (Total_Lines, 192)

    # 加载心率数据
    df_hr = pd.read_csv("hr_data5.txt", header=None)
    series_hr = df_hr[0].values.astype('float32')

    # 【关键】根据 process_new.py 的块结构进行重塑
    num_samples = len(data_matrix) // WINDOW_SIZE
    X = data_matrix[:num_samples * WINDOW_SIZE].reshape(num_samples, WINDOW_SIZE, 192)
    # 每 100 行对应一个真实的心率值
    Y = series_hr[:num_samples * WINDOW_SIZE : WINDOW_SIZE]

    # 简单归一化
    X_min, X_max = np.min(X), np.max(X)
    X = (X - X_min) / (X_max - X_min + 1e-7)

    print(f"Done. Samples: {X.shape[0]}, X_shape: {X.shape[1:]}, Y_shape: {Y.shape}")

except FileNotFoundError:
    print("错误: 找不到训练文件")
    exit()

# ==========================================
# 2. 模型构建 (Conv1D + LSTM)
# ==========================================
if CONTINUE_TRAINING_MODEL != "":
    model = keras.models.load_model(CONTINUE_TRAINING_MODEL)
else:
    main_input = keras.Input(shape=(WINDOW_SIZE, 192), name='main_input')

    # 新增: Conv1D 空间特征合成层
    x = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(main_input)
    x = keras.layers.MaxPooling1D(pool_size=2)(x) 

    # 原有的时序提取结构
    x = keras.layers.LSTM(64, return_sequences=True, name='lstm_1')(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.LSTM(32, name='lstm_2')(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(16, activation='relu', name='dense_1')(x)
    hr_output = keras.layers.Dense(1, name='hr_output')(x)

    model = keras.Model(inputs=main_input, outputs=hr_output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

model.summary()

# ==========================================
# 3. 开始训练
# ==========================================
early_stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

try:
    model.fit(X, Y,
              epochs=100,
              batch_size=BATCH_SIZE,
              validation_split=0.2, # 自动划分 20% 验证集
              shuffle=True,
              callbacks=[early_stop])
except KeyboardInterrupt:
    print("\n🛑 停止训练...")

model.save("csi_hr5.keras")

# ==========================================
# 4. 预测抽查 (Debug)
# ==========================================
print("\n🔍 预测抽查 (随机 10 个样本)")
test_idx = np.random.choice(len(X), 10)
preds = model.predict(X[test_idx], verbose=0).flatten()
y_true = Y[test_idx]

print(f"{'索引':<5} | {'真实心率':<10} | {'预测心率':<10} | {'误差':<10}")
for i in range(len(test_idx)):
    print(f"{i:<5} | {y_true[i]:<10.2f} | {preds[i]:<10.2f} | {abs(y_true[i]-preds[i]):<10.2f}")
