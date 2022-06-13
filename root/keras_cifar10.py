import time
start_time=time.time()
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
#import tensorflow as tf #mz
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model #mz
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam #mz
from tensorflow.keras.optimizers.schedules import InverseTimeDecay #mz
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pdb #mz pdb.set_trace() #mz

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,
help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# load the training and testing data, scale it into the range [0, 1],
# then reshape the design matrix
print("[INFO] loading CIFAR-10 data...")
#pdb.set_trace() #mz
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

print("[INFO] Training Size: {} Test Size: {}".format(trainX.shape[0],testX.shape[0])) #mz
len1=10000; len2=2000; #mz
trainX=trainX[0:len1] #mz
trainY=trainY[0:len1]#mz
testX=testX[0:len2] #mz
testY=testY[0:len2] #mz
print("[INFO] Modified Training Size: {} Test Size: {}".format(trainX.shape,testX.shape)) #mz

trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))
#pdb.set_trace() #mz
# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]

# Overfit Treatment
N_EPOCH=100
STEPS_PER_EPOCH = 313; #N_TRAIN//BATCH_SIZE;
lr_schedule = InverseTimeDecay(0.01, decay_steps=STEPS_PER_EPOCH*N_EPOCH, decay_rate=1, staircase=False);

model = Sequential([layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(3072,)), layers.Dropout(0.5), layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)), layers.Dropout(0.5), layers.Dense(10, activation='softmax')])

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(lr_schedule,momentum=0.5)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
epochs=N_EPOCH, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N_EPOCH), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N_EPOCH), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N_EPOCH), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N_EPOCH), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('res10_L4/fig_10k_100_sgd01-dec_m5_L0_l2d.png') #plt.savefig(args["output"])
total_time=(time.time()-start_time)/60;

#Saving:
model.save('res10_L4/model_10k_100_sgd01-dec_m5_L0_l2d')
np.save('res10_L4/hist_10k_100_sgd01-dec_m5_L0_l2d.npy',H.history)

print("[INFO] Execution Time: {}".format(total_time))
print(max(H.history['val_accuracy']))

