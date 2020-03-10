"""
print("\n#################### num_epochs = 4, batch_size = 128, learning_rate = 0.01 ####################")
num_epochs = 4
batch_size = 128
learning_rate = 0.01

print("\n********************* SGD *********************")

opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer= opt,
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2,validation_data=(x_test, y_test))

trainacc_SGD=history.history['accuracy']
trainloss_SGD=history.history['loss']
testacc_SGD=history.history['val_accuracy']
testloss_SGD=history.history['val_loss']


print("\n********************* Nesterov *********************")
# set the optimizer
opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=True)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer= opt,
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2,validation_data=(x_test, y_test))

trainacc_Nesterov=history.history['accuracy']
trainloss_Nesterov=history.history['loss']
testacc_Nesterov=history.history['val_accuracy']
testloss_Nesterov=history.history['val_loss']


print("\n********************* Adagrad *********************")
opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer= opt,
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2,validation_data=(x_test, y_test))

trainacc_Adagrad_64_01=history.history['accuracy']
trainloss_Adagrad=history.history['loss']
testacc_Adagrad=history.history['val_accuracy']
testloss_Adagrad=history.history['val_loss']
"""




###############Part1

name = ["trainacc","testacc","trainloss","testloss"]
opt = ["SGD","Nesterov","Adagrad"]
bs = ["64","128","256"]
lr = ["01","001","005"]
history = ['accuracy','val_accuracy','loss','val_loss']
color = ["black","darkorange","limegreen","b","r","y","c","m","slategray"]

# #############part2
"""

for j in range(4):
    print(name[j]+"_"+opt[0]+"_"+bs[0]+"_"+lr[0], "= history.history"+ "['" + history[j] + "']")
"""

# #############part3

i = [0,1,2]
it1 = iter(color)
it2 = iter(color)
namelist = []
a = ["abc"]
for i in opt:
    for j in bs:
        print("plt.plot("+name[0]+"_"+i+"_"+j+"_"+lr[0]+","+"'"+next(it1)+"'"+")")
        namelist.append(name[0] + "_" + i + "_" + j + "_" + lr[0])

for i in opt:
    for j in bs:
        print("plt.plot("+name[1]+"_"+i+"_"+j+"_"+lr[0]+","+"'--"+next(it2)+"'"+")")
        namelist.append(name[1] + "_" + i + "_" + j + "_" + lr[0])
print("legend list:")
print(namelist)


############### part 4
optn = ("SGD","Nesterov","Adagrad")
for i in range(3):
    name = "trainacc_"+optn[i]+"_64_01"
    locals()["trainacc_"+optn[i]+"_64_01"] = i
print (trainacc_SGD_64_01,trainacc_Nesterov_64_01,trainacc_Adagrad_64_01)

bs = ("64","128","256")
for i in range(3):
    name = "trainacc_SGD_"+bs[i]+"_01"
    locals()["trainacc_SGD_"+bs[i]+"_01"]=i
print(trainacc_SGD_64_01,trainacc_SGD_128_01,trainacc_SGD_256_01)

i=1
name = "trainacc_"+"SGD_"+bs[i]+"_01" # the name is "trainacc_SGD_64_01"
locals()["trainacc_"+"SGD"+bs[i]+"_01"] = i
print(trainacc_SGD_64_01)

"""