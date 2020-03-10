"""
MNIST dataset trained on a small network, different optimizers
"""

# install TensorFlow
import tensorflow as tf
import matplotlib.pyplot as plt

# prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


num_epochs = 3
learning_rate = 0.01
bs = ["64","128","256"]
opt = ["tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)",
       "tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=True)",
       "tf.keras.optimizers.Adagrad(learning_rate=0.01"]
optn = ["SGD","Nesterov","Adagrad"]

for i in range(3):
    print("\n************************** SGD , bs=",int(bs[i]), ", lr=0.01--start**************************")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=int(bs[i]), verbose=2,
                        validation_data=(x_test, y_test))
    name = "trainacc_"+"SGD_"+bs[i]+"_01" # the name is "trainacc_SGD_64_01"
    locals()["trainacc_"+"SGD_"+bs[i]+"_01"] = history.history['accuracy']
    name = "trainloss_"+"SGD_"+bs[i]+"_01"
    locals()["trainloss_"+"SGD_"+bs[i]+"_01"] = history.history['loss']
    name = "testacc_"+"SGD_"+bs[i]+"_01"
    locals()["testacc_"+"SGD_"+bs[i]+"_01"] = history.history['val_accuracy']
    name = "testloss_"+"SGD_"+bs[i]+"_01"
    locals()["testloss_"+"SGD_"+bs[i]+"_01"] = history.history['val_loss']
    print("\n************************** SGD , bs=",int(bs[i]), ", lr=0.01--end**************************")
    print("\n************************** Nesterov , bs=",int(bs[i]), ", lr=0.01--start**************************")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=True),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=int(bs[i]), verbose=2,
                        validation_data=(x_test, y_test))
    name = "trainacc_"+"Nesterov_"+bs[i]+"_01" # the name is "trainacc_SGD_64_01"
    locals()["trainacc_"+"Nesterov_"+bs[i]+"_01"] = history.history['accuracy']
    name = "trainloss_"+"Nesterov_"+bs[i]+"_01"
    locals()["trainloss_"+"Nesterov_"+bs[i]+"_01"] = history.history['loss']
    name = "testacc_"+"Nesterov_"+bs[i]+"_01"
    locals()["testacc_"+"Nesterov_"+bs[i]+"_01"] = history.history['val_accuracy']
    name = "testloss_"+"Nesterov_"+bs[i]+"_01"
    locals()["testloss_"+"Nesterov_"+bs[i]+"_01"] = history.history['val_loss']
    print("\n************************** Nesterov , bs=",int(bs[i]), ", lr=0.01--end**************************")
    print("\n************************** Adagrad , bs=",int(bs[i]), ", lr=0.01--start**************************")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=int(bs[i]), verbose=2,
                        validation_data=(x_test, y_test))
    name = "trainacc_"+"Adagrad_"+bs[i]+"_01" # the name is "trainacc_SGD_64_01"
    locals()["trainacc_"+"Adagrad_"+bs[i]+"_01"] = history.history['accuracy']
    name = "trainloss_"+"Adagrad_"+bs[i]+"_01"
    locals()["trainloss_"+"Adagrad_"+bs[i]+"_01"] = history.history['loss']
    name = "testacc_"+"Adagrad_"+bs[i]+"_01"
    locals()["testacc_"+"Adagrad_"+bs[i]+"_01"] = history.history['val_accuracy']
    name = "testloss_"+"Adagrad_"+bs[i]+"_01"
    locals()["testloss_"+"Adagrad_"+bs[i]+"_01"] = history.history['val_loss']
    print("\n************************** Adagrad , bs=",int(bs[i]), ", lr=0.01--end**************************")




#"""

###这段代码是能正常运行的，但是我还没加legend,也没加线条的颜色和形状
# bs = ["64","128","256"]
# optn = ["SGD","Nesterov","Adagrad"]
plt.plot(trainacc_SGD_64_01)
plt.plot(trainacc_Nesterov_64_01)
plt.plot(trainacc_Adagrad_64_01)
plt.plot(testacc_SGD_64_01)
plt.plot(testacc_Nesterov_64_01)
plt.plot(testacc_Adagrad_64_01)
plt.plot(trainacc_SGD_128_01)
plt.plot(trainacc_Nesterov_128_01)
plt.plot(trainacc_Adagrad_128_01)
plt.plot(testacc_SGD_128_01)
plt.plot(testacc_Nesterov_128_01)
plt.plot(testacc_Adagrad_128_01)
plt.plot(trainacc_SGD_256_01)
plt.plot(trainacc_Nesterov_256_01)
plt.plot(trainacc_Adagrad_256_01)
plt.plot(testacc_SGD_256_01)
plt.plot(testacc_Nesterov_256_01)
plt.plot(testacc_Adagrad_256_01)
plt.title('Accuracy-opt-bs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('accuracy-bs.png')
plt.show()

#"""



# ######################更新2020/3/10/18：37
# 原因已经找到：'--black'格式错误，plt.plot不能成功读取，须改为（color = "black", linestyle = "--"）
# 以下为原始错误代码

# 这段代码中是通过循环写出来的，加了legend和线条style，但是运行不出来
# plot and compare the accuracies of training set
"""
plt.plot(trainacc_SGD_64_01,'black')
plt.plot(trainacc_SGD_128_01,'darkorange')
plt.plot(trainacc_SGD_256_01,'limegreen')
plt.plot(trainacc_Nesterov_64_01,'b')
plt.plot(trainacc_Nesterov_128_01,'r')
plt.plot(trainacc_Nesterov_256_01,'y')
plt.plot(trainacc_Adagrad_64_01,'c')
plt.plot(trainacc_Adagrad_128_01,'m')
plt.plot(trainacc_Adagrad_256_01,'slategray')
plt.plot(testacc_SGD_64_01,'--black')
plt.plot(testacc_SGD_128_01,'--darkorange')
plt.plot(testacc_SGD_256_01,'--limegreen')
plt.plot(testacc_Nesterov_64_01,'--b')
plt.plot(testacc_Nesterov_128_01,'--r')
plt.plot(testacc_Nesterov_256_01,'--y')
plt.plot(testacc_Adagrad_64_01,'--c')
plt.plot(testacc_Adagrad_128_01,'--m')
plt.plot(testacc_Adagrad_256_01,'--slategray')

plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['trainacc_SGD_64_01', 'trainacc_SGD_128_01', 'trainacc_SGD_256_01',
            'trainacc_Nesterov_64_01', 'trainacc_Nesterov_128_01', 'trainacc_Nesterov_256_01',
            'trainacc_Adagrad_64_01', 'trainacc_Adagrad_128_01', 'trainacc_Adagrad_256_01',
            'testacc_SGD_64_01', 'testacc_SGD_128_01', 'testacc_SGD_256_01',
            'testacc_Nesterov_64_01', 'testacc_Nesterov_128_01', 'testacc_Nesterov_256_01',
            'testacc_Adagrad_64_01', 'testacc_Adagrad_128_01', 'testacc_Adagrad_256_01'],loc=0)
plt.savefig('accuracy.png')
plt.show()

"""


