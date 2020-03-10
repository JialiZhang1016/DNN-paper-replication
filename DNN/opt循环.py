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


print("\n#################### num_epochs = 4, batch_size = 64, learning_rate = 0.01 ####################")


num_epochs = 4
batch_size = 64
learning_rate = 0.01
opt = ["tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)",
       "tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=True)",
       "tf.keras.optimizers.Adagrad(learning_rate=0.01"]
optn = ["SGD","Nesterov","Adagrad"]
datan = ["trainacc_","trainloss_","testacc_","testloss_"]

for i in range(3):
    print("\n**************************", i, "--start**************************")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2,
                        validation_data=(x_test, y_test))
    name = "trainacc_"+optn[i]+"_64_01"
    locals()["trainacc_"+optn[i]+"_64_01"] = history.history['accuracy']
    name = "trainloss_"+optn[i]+"_64_01"
    locals()["trainloss_"+optn[i]+"_64_01"] = history.history['loss']
    name = "testacc_"+optn[i]+"_64_01"
    locals()["testacc_"+optn[i]+"_64_01"] = history.history['val_accuracy']
    name = "testloss_"+optn[i]+"_64_01"
    locals()["testloss_"+optn[i]+"_64_01"] = history.history['val_loss']
    print("\n**************************",i,"--end***************************")


# optn = ["SGD","Nesterov","Adagrad"]
# ********************test*********************
plt.plot(trainacc_SGD_64_01)
plt.plot(trainacc_Nesterov_64_01)
plt.plot(trainacc_Adagrad_64_01)
plt.plot(testacc_SGD_64_01)
plt.plot(testacc_Nesterov_64_01)
plt.plot(testacc_Adagrad_64_01)
plt.title('Training accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['SGD', 'Nesterov','Adagrad',"t1","t2","t3"], loc='lower right')
plt.savefig('training_accuracy.png')
plt.show()




"""



# plot and compare the accuracies of training set
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
