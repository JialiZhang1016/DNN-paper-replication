
import tensorflow as tf
import numpy as np
import pandas as pd
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


num_epochs = 4
batch_size = [64,128,256]
learning_rate = 0.01
model = MLP()
data_loader = MNISTLoader()
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
M = []
df = pd.DataFrame(columns=('epoch','loss','accuracy','Megan','batch_size'))


for bs in batch_size:
    y_pred = model(X)
    losss = []
    # 由图像可以看出，第一层循环之后，数据出现错误，模型是在之前模型的基础上算的，而不是一个新的batch_size。尝试改进
    # 不行
    # 尝试改model！！！
    # 不行
    # 加入losss = []
    for epoch_index in range(num_epochs):
        num_batch = int(data_loader.num_train_data // bs)
        for batch_index in range(num_batch):
            X, y = data_loader.get_batch(bs)
            with tf.GradientTape() as tape:
                y_pred = model(X)
                losss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
                loss = tf.reduce_mean(losss) 
                sparse_categorical_accuracy.update_state(y_true=y, y_pred=y_pred)
            grads = tape.gradient(loss, model.variables)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))

            con1 = np.array(grads[0])    # TensorShape:2 (784*100)
            lay1 = np.array(grads[1])    # TensorShape:1 (100,)
            con2 = np.array(grads[2])    # TensorShape:2 (100*10)
            lay2 = np.array(grads[3])    # TensorShape:1 (10,)
            sumcon1 = np.sum(con1 ** 2)
            sumlay1 = np.sum(lay1 ** 2)
            sumcon2 = np.sum(con2 ** 2)
            sumlay2 = np.sum(con2 ** 2)
            M.append(sumcon1 + sumcon2 + sumlay1 + sumlay2)

        Megan = tf.reduce_mean(M)
        accuracy = sparse_categorical_accuracy.result()
        df.loc[df.shape[0]+1] = [epoch_index, loss.numpy(), accuracy.numpy(), Megan.numpy(),bs]
        print("epoch %d: loss: %f, accuracy: %f, Megan: %f, batch_size: %d" % (epoch_index, loss.numpy(), accuracy.numpy(), Megan.numpy(), bs))

df.to_csv('Megan_bs.csv')




bs1, = plt.plot(df[df.batch_size == 64].epoch, df[df.batch_size ==  64].Megan)
bs2, = plt.plot(df[df.batch_size == 128].epoch, df[df.batch_size == 128].Megan)
bs3, = plt.plot(df[df.batch_size == 256].epoch, df[df.batch_size == 256].Megan)
plt.legend([bs1,bs2,bs3],['bs=64','bs=128','bs=256'], loc='lower right')
plt.title("Megan_bs[64,128,256]")
plt.ylabel('Megan')
plt.xlabel('Epoch')
plt.savefig('Megan_bs[64,128,256].png')
plt.show()

bs1, = plt.plot(df[df.batch_size == 64].epoch, df[df.batch_size ==  64].loss)
bs2, = plt.plot(df[df.batch_size == 128].epoch, df[df.batch_size == 128].loss)
bs3, = plt.plot(df[df.batch_size == 256].epoch, df[df.batch_size == 256].loss)
plt.legend([bs1,bs2,bs3],['bs=64','bs=128','bs=256'], loc='lower right')
plt.title("loss_bs[64,128,256]")
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.savefig('loss_bs[64,128,256].png')
plt.show()

bs1, = plt.plot(df[df.batch_size ==  64].epoch, df[df.batch_size ==  64].accuracy)
bs2, = plt.plot(df[df.batch_size == 128].epoch, df[df.batch_size == 128].accuracy)
bs3, = plt.plot(df[df.batch_size == 256].epoch, df[df.batch_size == 256].accuracy)
plt.legend([bs1,bs2,bs3],['bs=64','bs=128','bs=256'], loc='lower right')
plt.title("accuracy_bs[64,128,256]")
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.savefig('accuracy_bs[64,128,256].png')
plt.show()
