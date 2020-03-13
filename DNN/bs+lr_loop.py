import tensorflow as tf
import matplotlib.pyplot as plt

# prepare the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# set the parameters
num_epochs = 10
learning_rate = [0.01,0.001,0.005]
batch_size = [64,128,256]
opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False)
trainacc_set = []
trainloss_set = []
testacc_set = []
testloss_set = []

for lr in learning_rate:
    for bs in batch_size:
        # key refers to the optimizer name. This parameter is changeable.
        # bs refers to the batch_size. This parameter is changeable.
        # lr refers to the learning_rate. This parameter is fixed 0.01.
        print("\n************************** batch_size =",bs,", learning_rate =",lr,"--start **************************")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=bs, verbose=2,  # 这里要把bs的类型强制转换成int
                            validation_data=(x_test, y_test))

        trainacc = []
        trainacc.append('lr=')
        trainacc.append(lr)
        trainacc.append(',bs=')
        trainacc.append(bs)
        trainacc.append(history.history['accuracy'])


        trainloss = []
        trainloss.append('lr=')
        trainloss.append(lr)
        trainloss.append(',bs=')
        trainloss.append(bs)
        trainloss.append(history.history['loss'])

        testacc = []
        testacc.append('lr=')
        testacc.append(lr)
        testacc.append(',bs=')
        testacc.append(bs)
        testacc.append(history.history['val_accuracy'])

        testloss = []
        testloss.append('lr=')
        testloss.append(lr)
        testloss.append(',bs=')
        testloss.append(bs)
        testloss.append(history.history['val_loss'])

        trainacc_set.append(trainacc)
        trainloss_set.append(trainloss)
        testacc_set.append(testacc)
        testloss_set.append(testloss)

index = 0
trainloss_name = []
for i in range(9):
    trainloss_name.append(trainacc_set[index][0]+str(trainacc_set[index][1])+trainacc_set[index][2]+str(trainacc_set[index][3]))
    trainloss_result = []
    trainloss_result = trainloss_set[index][4]
    plt.plot(trainloss_result)
    index = index + 1


plt.title('Training Loss_lr=' + str(learning_rate) + '_bs=' + str(batch_size))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(trainloss_name, loc='upper right')
plt.savefig('training_loss_lr=' + str(learning_rate) + '_bs=' + str(batch_size) + '.png')
plt.show()
