import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import datetime
from sklearn.model_selection import train_test_split # 新增：用于科学打乱数据

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
# 【修复 1】：减小批次，让模型多更新几次，增强泛化能力
BATCH_SIZE = 128  
WINDOW_SIZE = 100

# ==========================================
# 1. 读取数据 & 结构化重塑
# ==========================================
print(f"Reading data ({datetime.datetime.now()})...")

try:
    X = np.load("training_data.npy")
    Y = np.load("hr_data.npy")

    # 对 X 进行样本级别的 Z-score 标准化
    X_mean = np.mean(X, axis=(1, 2), keepdims=True)
    X_std = np.std(X, axis=(1, 2), keepdims=True)
    X = (X - X_mean) / (X_std + 1e-7)

    # 对 Y (心率标签) 也进行标准化
    Y_mean = np.mean(Y)
    Y_std = np.std(Y)
    Y_norm = (Y - Y_mean) / (Y_std + 1e-7)

    # 【修复 2】：使用 sklearn 彻底打乱数据并切分！(关键所在)
    print("Splitting data randomly...")
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y_norm, test_size=0.2, random_state=42
    )

    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

except FileNotFoundError:
    print("错误: 找不到文件")
    exit()

# ==========================================
# 2. 模型构建 (Conv1D + LSTM)
# ==========================================
if CONTINUE_TRAINING_MODEL != "":
    model = keras.models.load_model(CONTINUE_TRAINING_MODEL)
else:
    main_input = keras.Input(shape=(WINDOW_SIZE, 192), name='main_input')

    x = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(main_input)
    x = keras.layers.MaxPooling1D(pool_size=2)(x) 
    # 【修复 3】：增加 Dropout，防止死记硬背
    x = keras.layers.Dropout(0.3)(x) 

    x = keras.layers.LSTM(64, return_sequences=True, name='lstm_1')(x)
    x = keras.layers.Dropout(0.3)(x) 

    x = keras.layers.LSTM(32, name='lstm_2')(x)
    x = keras.layers.Dropout(0.3)(x)

    # 【修复 4】：增加 L2 正则化惩罚，限制权重过大
    x = keras.layers.Dense(16, activation='elu', kernel_regularizer=keras.regularizers.l2(0.01), name='dense_1')(x)
    hr_output = keras.layers.Dense(1, name='hr_output')(x)

    model = keras.Model(inputs=main_input, outputs=hr_output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# ==========================================
# 3. 开始训练
# ==========================================
try:
    # 【修复 5】：传入我们刚才彻底打乱好的 X_val 和 Y_val
    model.fit(X_train, Y_train,
              epochs=100,
              batch_size=BATCH_SIZE,
              validation_data=(X_val, Y_val), 
              shuffle=True)
except KeyboardInterrupt:
    print("\n🛑 停止训练...")

model.save("csi_hr5.keras")

# ==========================================
# 4. 预测抽查 (Debug)
# ==========================================
print("\n🔍 预测抽查 (验证集随机 10 个样本)")
# 从验证集中抽查，这样更客观
test_idx = np.random.choice(len(X_val), 10)

preds_norm = model.predict(X_val[test_idx], verbose=0).flatten()

# 还原预测值
preds = preds_norm * Y_std + Y_mean
# 还原真实标签
y_true = Y_val[test_idx] * Y_std + Y_mean  

print(f"{'索引':<5} | {'真实心率':<10} | {'预测心率':<10} | {'误差':<10}")
for i in range(len(test_idx)):
    print(f"{i:<5} | {y_true[i]:<10.2f} | {preds[i]:<10.2f} | {abs(y_true[i]-preds[i]):<10.2f}")