# from __future__ import absolute_import, division, print_function, unicode_literals

# 安装 TensorFlow
import tensorflow as tf
import matplotlib.pyplot as plt


# 准备数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 使用Keras建立模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 模型参数设定
model.compile(optimizer= tf.keras.optimizers.SGD(lr=0.1),
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 模型拟合与验证
history = model.fit(x_train, y_train, epochs=160, batch_size=64,verbose=2,validation_data=(x_test, y_test))

# print(history.history.keys())查看函数返回值
print(history.history.keys())
'''
训练过程中的四个值['accuracy', 'loss', 'val_accuracy', 'val_loss']
['accuracy'] refers to the accuracy of training set
['val_accuracy'] refers to the accuracy of validation set
['loss'] refers to the loss of training set
['val_loss'] refers to the loss of validation set
'''

# plot the accuracy of training set and validation set
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# plot the loss of training set and validation set
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

