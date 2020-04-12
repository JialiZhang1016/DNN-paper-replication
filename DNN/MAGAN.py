
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output

# 参数设置
num_epochs = 10
batch_size = 128
learning_rate = 0.01

model = MLP()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
num_batch = int(data_loader.num_train_data // batch_size) # 468
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
loss_set = []
accuracy_set = []
Megan_set = []
M = []

for epoch_index in range(num_epochs):                 # 第一层循环，epoches
    for batch_index in range(num_batch):              # 第二层循环，mini_batch
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            losses = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)    # 得到交叉熵
            loss = tf.reduce_mean(losses)                                                        # losses——TensorShape:1, 128 (=batch_size)
            sparse_categorical_accuracy.update_state(y_true=y, y_pred=y_pred)                    # 更新mini_batch里面的Accuracy
        grads = tape.gradient(loss, model.variables)                                             # 获取梯度
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))
# 求norm of grads. the grads has 4 lists, named con1,lay2,con2,lay3
        con1 = np.array(grads[0])    # TensorShape:2 (784*100) 第一层与第二层的连接
        lay2 = np.array(grads[1])    # TensorShape:1 (100,)    第二层的变量
        con2 = np.array(grads[2])    # TensorShape:2 (100*10)  第二层与第三层的连接
        lay3 = np.array(grads[3])    # TensorShape:1 (10,)     第三层的变量
        sumcon1 = np.sum(con1 ** 2)
        sumlay2 = np.sum(lay2 ** 2)
        sumcon2 = np.sum(con2 ** 2)
        sumlay3 = np.sum(lay3 ** 2)
        M.append(sumcon1 + sumcon2 + sumlay2 + sumlay3)    #2-norm of the grads

    Megan = tf.reduce_mean(M)
    accuracy = sparse_categorical_accuracy.result()
    # 分别求出Megan, accuracy, loss的列表
    Megan_set.append(Megan)
    accuracy_set.append(accuracy)
    loss_set.append(loss)
    print("epoch %d: loss %f accuracy: %f Megan: %f " % (epoch_index, loss.numpy(), accuracy, Megan))

plt.plot(Megan_set)
plt.title("Megan")
plt.ylabel('Megan')
plt.xlabel('Epoch')
plt.savefig('Megan.png')
plt.show()

plt.plot(accuracy_set)
plt.title("accuracy")
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.savefig('accuracy.png')


plt.plot(loss_set)
plt.title("loss")
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.savefig('loss.png')
