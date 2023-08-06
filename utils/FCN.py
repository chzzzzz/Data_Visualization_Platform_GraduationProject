# FCN全卷积
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,load_model
from keras.utils import np_utils
from keras.layers import (Input, Reshape)
import tensorflow.keras.layers
import numpy as np
import pandas as pd
import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.datasets.mnist as mnist

# model建立时读取有标签数据
def readucr(filename):
    # data = np.loadtxt(filename, delimiter=',')
    data = pd.read_excel(filename)
    # Y第一列是标签
    data = data.values
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y
# model调用时读取无标签
def read_test(filename):
    # data = np.loadtxt(filename, delimiter=',')
    data = pd.read_excel(filename)
    data = data.values
    X = data[:, :]
    return X
# 调用fcn模型
def use_fcn(path):
    x_test = read_test(path)
    x_test = x_test[1:,1:]
    x_test_mean = x_test.mean()
    x_test_std = x_test.std()
    x_test = (x_test - x_test_mean) / (x_test_std)
    x_test = x_test.reshape(x_test.shape + (1, 1,))
    my_model = load_model('./ckpt_dir/FCN_1/cnn_2.h5')
    # (train_image, train_label), (test_image, test_label) = mnist.load_data()
    # test_image = np.expand_dims(test_image, axis=-1)p
    # my_model.evaluate(test_image, test_label)
    result = my_model.predict(x_test.astype(float))
    print("---------------------result---------------------",'\n',result)
    return result
# 建模型语句绝对纠结路径
def built_model():
    nb_epochs = 1499

    flist = ['gf']
    for each in flist:
        fname = each
        x_train, y_train = readucr('D:/Essay/fcn/train_xy.xlsx')
        x_test, y_test = readucr('D:/Essay/fcn/test_xy.xlsx')
        nb_classes = len(np.unique(y_test))
        batch_size = min(x_train.shape[0] / 10, 16)
        print("x_train.shape[0]", x_train.shape[0])
        # y转化成0:nb_classes - 1范围
        y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
        y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)
        # np_utils.to_categorical 将y_train转化为one_hot编码
        # >> > a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
        # >> > a = tf.constant(a, shape=[4, 4])
        # >> > print(a)
        # tf.Tensor(
        #     [[1. 0. 0. 0.]
        #      [0. 1. 0. 0.]
        #     [0. 0. 1. 0.]
        # [0. 0. 0. 1.]], shape = (4, 4), dtype = float32)
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        x_train_mean = x_train.mean()
        x_train_std = x_train.std()
        x_train = (x_train - x_train_mean) / (x_train_std)
        x_test_mean = x_test.mean()
        x_test_std = x_test.std()
        x_test = (x_test - x_test_mean) / (x_test_std)
        x_train = x_train.reshape(x_train.shape + (1, 1,))
        x_test = x_test.reshape(x_test.shape + (1, 1,))
        # x_train.shape: (895, 6, 1, 1)
        # x_train.shape[1:] (6, 1, 1)
        x = tf.keras.layers.Input([6,1,1])
        #    drop_out = Dropout(0.2)(x)
        # border_mode补零吗?same是全0填充，valid不填充
        #model = Sequential([Conv2D(128, 8,1, padding='same', activation='relu', input_shape=(6,1,1))])

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(128,8, strides=(1, 1), padding='same',input_shape=[6,1,1]))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Conv2D(256,5, strides=(1, 1), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Conv2D(128,3, strides=(1, 1), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        # conv1 = tf.keras.layers.Conv2D(128, 8, 1, padding='same')(x)
        # conv1 = keras.layers.normalization.BatchNormalization()(conv1)
        # conv1 = keras.layers.Activation('relu')(conv1)
        #
        # #    drop_out = Dropout(0.2)(conv1)
        # conv2 = keras.layers.Conv2D(256, 5, 1, padding='same')(conv1)
        # conv2 = keras.layers.normalization.BatchNormalization()(conv2)
        # conv2 = keras.layers.Activation('relu')(conv2)
        #
        # #    drop_out = Dropout(0.2)(conv2)
        # conv3 = keras.layers.Conv2D(128, 3, 1, padding='same')(conv2)
        # conv3 = keras.layers.normalization.BatchNormalization()(conv3)
        # conv3 = keras.layers.Activation('relu')(conv3)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))

        # full = keras.layers.pooling.GlobalAveragePooling2D()(conv3)

        # out = keras.layers.Dense(nb_classes, activation='softmax')(full)

        # model = Model(input=x, output=out)

        # optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                      patience=50, min_lr=0.0001)
        history = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
                            verbose=1, validation_data=(x_test, Y_test), callbacks=[reduce_lr])
        model.save('../ckpt_dir/FCN_1/cnn_2.h5')
        # saver = tf.train.Saver(history)
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # saver.save(sess, '../ckpt_dir/FCN_1')
    def predict(model,path):
        data = pd.read_excel(path)
        # Y第一列是标签
        data = data.values
        true_label = data[1:, 0]
        x_test = data[1:, 1:]
        x_test_mean = x_test.mean()
        x_test_std = x_test.std()
        x_test = (x_test - x_test_mean) / (x_test_std)
        x_test = x_test.reshape(x_test.shape + (1, 1,))
        pre_label = model.predict(x_test.astype(float))
        pre_label = pre_label.tolist()
        pre_result = np.array(pre_label)
        index_max = np.argmax(pre_result, axis=1)
        # 返回label处于0-3，处理成1-4
        index_max = (index_max + 1).tolist()
        label = index_max
        labels = {1: '正常', 2: '全部遮挡', 3: '部分遮挡',
                  4: '断路'}
        print(classification_report(label, true_label,
                                    target_names=[l for l in labels.values()]))

        conf_mat = confusion_matrix(label, true_label)
        fig = plt.figure(figsize=(6, 6))
        width = np.shape(conf_mat)[1]
        height = np.shape(conf_mat)[0]

        res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
        for i, row in enumerate(conf_mat):
            for j, c in enumerate(row):
                if c > 0:
                    plt.text(j - .2, i + .1, c, fontsize=16)
        plt.rcParams['axes.grid'] = False
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        cb = fig.colorbar(res)
        plt.title('Confusion Matrix')
        _ = plt.xticks(range(4), [l for l in labels.values()], rotation=90)
        _ = plt.yticks(range(4), [l for l in labels.values()])
        plt.show()

    predict(model,'D:/Essay/fcn/test_xy.xlsx')
    acc = history.history['accuracy']
    # 历史验证准确率
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    # 历史验证误差
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
if __name__ == '__main__':
    built_model()



