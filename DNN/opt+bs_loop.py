import tensorflow as tf
import matplotlib.pyplot as plt

# prepare the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# set the parameters
num_epochs = 3
learning_rate = 0.01
bs = ["64","128","256"]
opt = {
    "SGD":tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
    "Nesterov":tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=True),
    "Adagrad":tf.keras.optimizers.Adagrad(learning_rate=0.01)}
legendacc= []
legendloss = []


"""

这个循环有两层
外层为optimizer的循环，一共有3种优化器可供选择["SGD","Nesterov","Adagrad"]
内层为batch_size的循环，取值为[64,128,256]
固定learning_rate，为0.01
固定epochs,为16

"""

for (key,value) in opt.items():
    for i in range(3):
        # key refers to the optimizer name. This parameter is changeable.
        # bs refers to the batch_size. This parameter is changeable.
        # lr refers to the learning_rate. This parameter is fixed 0.01.
        print("\n**************************",key,", bs=", int(bs[i]), ", lr=0.01 --start **************************")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=value,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=int(bs[i]), verbose=2,  # 这里要把bs的类型强制转换成int
                            validation_data=(x_test, y_test))
        """
        设置循环变量，并且为其赋值,语句：
        name = "self_def_name" 
        locals()["self_def_name"]= value
        """
        name =   "trainacc_" + key +"_" + bs[i] + "_01"  # 这里的bs[i]类型为"str". The first one's name is "trainacc_SGD_64_01"
        locals()["trainacc_" + key +"_" + bs[i] + "_01"] = history.history['accuracy']
        name =   "trainloss_"+ key +"_" + bs[i] + "_01"
        locals()["trainloss_"+ key +"_" + bs[i] + "_01"] = history.history['loss']
        name =   "testacc_"  + key +"_" + bs[i] + "_01"
        locals()["testacc_"  + key +"_" + bs[i] + "_01"] = history.history['val_accuracy']
        name =   "testloss_" + key +"_" + bs[i] + "_01"
        locals()["testloss_" + key +"_" + bs[i] + "_01"] = history.history['val_loss']
        # save the legend name of "Training accuracy" and "Testing accuracy" to the "legendacc" list.
        # save the legend name of "Training loss" and "Testing loss" to the "legendloss" list.
        legendacc.append ("trainacc_" + key +"_" + bs[i] + "_01")
        legendloss.append("trainloss_"+ key +"_" + bs[i] + "_01")
        legendacc.append ("testacc_"  + key +"_" + bs[i] + "_01")
        legendloss.append("testloss_" + key +"_" + bs[i] + "_01")
        print("\n**************************",key,", bs=", int(bs[i]), ", lr=0.01 --end **************************")
print(legendacc,legendloss)


"""
The following plt.plot() sentences can also be printed using circle function, which shows in part3 of dratf.py
"""

# plot the Train-Test Accuracy on (3-opt,3-bs, 1-lr)
# 下面的红色波浪线表示该变量未被定义，但在循环体中已经被定义且赋值，所以能够运行成功
plt.plot(trainacc_SGD_64_01, color = 'black')
plt.plot(testacc_SGD_64_01,linestyle='--', color='black')
plt.plot(trainacc_SGD_128_01, color = 'darkorange')
plt.plot(testacc_SGD_128_01,linestyle='--', color='darkorange')
plt.plot(trainacc_SGD_256_01, color = 'limegreen')
plt.plot(testacc_SGD_256_01,linestyle='--', color='limegreen')
plt.plot(trainacc_Nesterov_64_01, color = 'slategray')
plt.plot(testacc_Nesterov_64_01,linestyle='--', color='slategray')
plt.plot(trainacc_Nesterov_128_01, color = 'b')
plt.plot(testacc_Nesterov_128_01,linestyle='--', color='b')
plt.plot(trainacc_Nesterov_256_01, color = 'r')
plt.plot(testacc_Nesterov_256_01,linestyle='--', color='r')
plt.plot(trainacc_Adagrad_64_01, color = 'y')
plt.plot(testacc_Adagrad_64_01,linestyle='--', color='y')
plt.plot(trainacc_Adagrad_128_01, color = 'c')
plt.plot(testacc_Adagrad_128_01,linestyle='--', color='c')
plt.plot(trainacc_Adagrad_256_01, color = 'm')
plt.plot(testacc_Adagrad_256_01,linestyle='--', color='m')
plt.title('Train-Test Accuracy (3-opt,3-bs, 1-lr)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(legendacc,loc=0)
plt.savefig('Train-Test Accuracy (3-opt,3-bs, 1-lr).png')
plt.show()


plt.plot(trainloss_SGD_64_01, color = 'black')
plt.plot(testloss_SGD_64_01,linestyle='--', color='black')
plt.plot(trainloss_SGD_128_01, color = 'darkorange')
plt.plot(testloss_SGD_128_01,linestyle='--', color='darkorange')
plt.plot(trainloss_SGD_256_01, color = 'limegreen')
plt.plot(testloss_SGD_256_01,linestyle='--', color='limegreen')
plt.plot(trainloss_Nesterov_64_01, color = 'slategray')
plt.plot(testloss_Nesterov_64_01,linestyle='--', color='slategray')
plt.plot(trainloss_Nesterov_128_01, color = 'b')
plt.plot(testloss_Nesterov_128_01,linestyle='--', color='b')
plt.plot(trainloss_Nesterov_256_01, color = 'r')
plt.plot(testloss_Nesterov_256_01,linestyle='--', color='r')
plt.plot(trainloss_Adagrad_64_01, color = 'y')
plt.plot(testloss_Adagrad_64_01,linestyle='--', color='y')
plt.plot(trainloss_Adagrad_128_01, color = 'c')
plt.plot(testloss_Adagrad_128_01,linestyle='--', color='c')
plt.plot(trainloss_Adagrad_256_01, color = 'm')
plt.plot(testloss_Adagrad_256_01,linestyle='--', color='m')
plt.title('Train-Test Loss (3-opt,3-bs, 1-lr)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(legendloss,loc=0)
plt.savefig('Train-Test Loss (3-opt,3-bs, 1-lr).png')
plt.show()