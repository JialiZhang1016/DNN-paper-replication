# from __future__ import absolute_import, division, print_function, unicode_literals

# 安装 TensorFlow
import tensorflow as tf

mnist = tf.keras.datasets.mnist

# 准备数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

opt = tf.keras.optimizers.SGD(lr=0.1)

model.compile(optimizer= opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=16, batch_size=64,verbose=2)

model.evaluate(x_test,  y_test, verbose=2)
