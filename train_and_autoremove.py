'''
  Tensorflow 2.1 + Python3.6環境にて動作確認
'''
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam, SGD


'''
直近のCheckpointを履歴保持／削除するクラス
'''
class CheckpointTools(Callback):
    def __init__(self, save_best_only=True, num_saves=3):
        self.last_val_loss = float("inf") # save_best_only判定用
        self.save_best_only = save_best_only
        assert num_saves >= 1
        self.num_saves = num_saves     # 最大保存数(この数を超えたら最古を消す)
        self.recent_files = []         # ファイル履歴

    def remove_oldest_file(self):
        if len(self.recent_files) > self.num_saves:
            file_name = self.recent_files.pop(0)  # 先頭ファイルパス取得
            if os.path.exists(file_name):
                os.remove(file_name)              # ファイル削除
                print('-> remove:'+file_name)

    # 毎epoch 終了時に呼び出されるCallback
    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs['val_loss']

        # ModelCheckpointのファイル名に合わせる
        # ※epoch=(epoch+1) に注意
        file_name = os.path.join(CP_DIR, 'epoch{epoch:03d}-{val_loss:.5f}.hdf5').format(epoch=(epoch+1), val_loss=val_loss)

        if self.save_best_only:
            if val_loss < self.last_val_loss:
                print('-> store:'+file_name)
                self.last_val_loss = val_loss
                self.recent_files.append(file_name)
                self.remove_oldest_file()
        else:
            # ファイル履歴追加
            self.recent_files.append(file_name)
            print('-> store:'+file_name)
            # 古いファイル削除
            self.remove_oldest_file()

# --- Parameter Setting ---

# 最大エポック数
MAX_EPOCH = 100

# 打ち切り判断数
ES_PATIENCE = 10
# SaveBestOnly
SAVE_BEST_ONLY = True

# バッチサイズ
BATCH_SIZE = 16

# 入力データサイズ
IMAGE_W = 32
IMAGE_H = 32

# 分類カテゴリ数
N_CATEGORIES = 10

# 初期学習率
LEARN_RATE = 0.001

# チェックポイント保存フォルダ
CP_DIR = 'checkpoint'
os.makedirs(CP_DIR, exist_ok=True)

# --- Model Setting ---

## CNN定義 簡単な分類ならこれぐらいで可能
def Conv_L4(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_initializer='random_uniform', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='random_uniform', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='random_uniform', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='random_uniform', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    return model


# Input layer 作成
input_tensor = Input(shape=(IMAGE_H, IMAGE_W, 3))

# Base Model 作成
base_model = Conv_L4((IMAGE_H, IMAGE_W, 3))

# Output layer 作成
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(N_CATEGORIES, activation='softmax')(x)

# 全体Model作成
model = Model(inputs=base_model.input, outputs=predictions)

# Optimizer 選択
model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.8), loss='categorical_crossentropy',metrics=['accuracy'])

# Summary表示
model.summary()


# --- Train Setting ---

# 学習セットのロード(CIFAR-10)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 正規化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# One-Hot Vector化
y_train = keras.utils.to_categorical(y_train, N_CATEGORIES)
y_test = keras.utils.to_categorical(y_test, N_CATEGORIES)

# Callback選択
cb_funcs = []

# Checkpoint作成設定
check_point = ModelCheckpoint(filepath = os.path.join(CP_DIR, 'epoch{epoch:03d}-{val_loss:.5f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=SAVE_BEST_ONLY, mode='auto')
cb_funcs.append(check_point)

# 上で設定したCheckpointToolsをCallbackに組み込む
cb_cptools = CheckpointTools(save_best_only=SAVE_BEST_ONLY, num_saves=3)
cb_funcs.append(cb_cptools)

# Early-stopping Callback設定
if ES_PATIENCE >= 0:
    early_stopping = EarlyStopping(patience=ES_PATIENCE, verbose=1)
    cb_funcs.append(early_stopping)


# モデル訓練実行
history = model.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=MAX_EPOCH,
            validation_data=(x_test, y_test),
            callbacks=cb_funcs)

