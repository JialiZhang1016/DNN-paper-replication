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

# prepare the parameters
num_epochs = 3
batch_size = 64
learning_rate = 0.01
opt = {
    "SGD":tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
    "Nesterov":tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=True),
    "Adagrad":tf.keras.optimizers.Adagrad(learning_rate=0.01)}


for (key,value) in opt.items():
    print("\n**************************", key, "--start**************************")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=value,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2,
                        validation_data=(x_test, y_test))
    name = "trainacc_"+key+"_64_01"
    locals()["trainacc_"+key+"_64_01"] = history.history['accuracy']
    name = "trainloss_"+key+"_64_01"
    locals()["trainloss_"+key+"_64_01"] = history.history['loss']
    name = "testacc_"+key+"_64_01"
    locals()["testacc_"+key+"_64_01"] = history.history['val_accuracy']
    name = "testloss_"+key+"_64_01"
    locals()["testloss_"+key+"_64_01"] = history.history['val_loss']
    print("\n**************************",key,"--end***************************")



plt.plot(trainacc_SGD_64_01,"b")
plt.plot(trainacc_Nesterov_64_01,"r")
plt.plot(trainacc_Adagrad_64_01,"y")
plt.plot(testacc_SGD_64_01,"--b")
plt.plot(testacc_Nesterov_64_01,"--r")
plt.plot(testacc_Adagrad_64_01,"--y")
plt.title('Training accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['SGD', 'Nesterov','Adagrad',"T-SGD","T-Nesterov","T-Adagrad"], loc='lower right')
plt.savefig('accuracy-opt.png')
plt.show()




