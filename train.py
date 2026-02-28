import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import datetime


MSE_THRESHOLD = 0.01
CONTINUE_TRAINING_MODEL = ""


class stopCallback(keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_loss') <= MSE_THRESHOLD) or os.path.isfile("training.stop"):   
            print("\nReached {0} MSE; stopping training.".format(MSE_THRESHOLD)) 
            self.model.stop_training = True


data = []
data_hr = []
train_x = []
train_y = []

print ("Reading in training data ({0})...".format(datetime.datetime.now()))

with open("training_data.txt", "r") as f:
    for line in f:
        if line.strip() == "":
            continue
        pieces = line.strip().split(",")
        data.append(pieces)

with open("hr_data.txt", "r") as f:
    for line in f:
        if line.strip() == "":
            continue
        data_hr.append(float(line.strip()))

print("Done reading in game data ({0}).".format(datetime.datetime.now()))

if CONTINUE_TRAINING_MODEL != "":
    print("Loading saved model: {0}".format(CONTINUE_TRAINING_MODEL))
    model = keras.models.load_model(CONTINUE_TRAINING_MODEL)

else:
    # Build the model.
    ''' 
    layer(参数)(输入数据)
    如keras.layers.LSTM(64, return_sequences=True, name='lstm_1')(main_input)创建了一个具体的层,
    配置了神经元数量等参数,
    并将main_input作为输入,返回一个新的张量layers,
    创建一个具有64个神经元的LSTM层,
    return_sequences=True该层会输出每个时间步的结果,用于连接下一层LSTM
    keras.layers.Dropout(0.2,name='dropout_1')(layers)
    创建一个Dropout层,随机丢弃20%的神经元,防止过拟合
    接收上一行产生的layers变量作为输入
    处理后的结果再次更新给layers变量
    '''
    main_input = keras.Input(shape=(100, 192), name='main_input')   # main_input和layers调用创建的层对象，3x64=192
    layers = keras.layers.LSTM(64, return_sequences=True, name='lstm_1')(main_input)
    layers = keras.layers.Dropout(0.2, name='dropout_1')(layers)
    layers = keras.layers.LSTM(32, name='lstm_2')(layers)
    layers = keras.layers.Dropout(0.2, name='dropout_2')(layers)
    layers = keras.layers.Dense(16, activation='relu', name='dense_1')(layers)
    hr_output = keras.layers.Dense(1, name='hr_output')(layers)
   
    model = keras.Model(inputs=main_input, outputs=hr_output)

    # Compile the model.
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, 
        loss={'hr_output': 'mse'}
    )

# Print the model summary.
model.summary()

# Prepare the training data.
data = np.array(data, dtype=np.float32)

WINDOW_SIZE = 100
for i in range(len(data) - WINDOW_SIZE):
    train_x.append(data[i:i+WINDOW_SIZE])     # shape: (100, 192)
    # Get the average heart rate over the entire window.
    avg = sum(data_hr[i:i+WINDOW_SIZE]) / WINDOW_SIZE
    train_y.append(avg)  # shape: (1,)

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

print("Training data X shape: {0}".format(train_x.shape))
print("Training data Y shape: {0}".format(train_y.shape))

# Train the model.
callbacks_list = [stopCallback()]

model.fit(train_x, train_y, batch_size=32, epochs=5000, verbose=2, validation_split=0.2, callbacks=callbacks_list)
model.save("csi_hr.keras")
print("Model training complete!")
