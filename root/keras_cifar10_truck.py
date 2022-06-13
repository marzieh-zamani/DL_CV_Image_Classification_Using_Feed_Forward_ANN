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

testX0=np.copy(testX)
testY0=np.copy(testY)

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]

org=np.load('trucks/trucks_org.npy').astype("float") / 255.0
blur=np.load('trucks/trucks_blur.npy').astype("float") / 255.0
flipH=np.load('trucks/trucks_flipH.npy').astype("float") / 255.0
flipV=np.load('trucks/trucks_flipV.npy').astype("float") / 255.0
rotate90=np.load('trucks/trucks_rotate90.npy').astype("float") / 255.0
rotate180=np.load('trucks/trucks_rotate180.npy').astype("float") / 255.0
scale=np.load('trucks/trucks_scale.npy').astype("float") / 255.0
truckY=np.load('trucks/truckY.npy')
truckY = lb.transform(truckY)


testX0=testX0[np.logical_not(testY0[:,0]==9)]
testY0=testY0[np.logical_not(testY0[:,0]==9)]
testY0 = lb.transform(testY0)

dataY=np.zeros([org.shape[0]+testX0.shape[0],testY0.shape[1]])
dataY[0:org.shape[0]]=truckY
dataY[org.shape[0]:dataY.shape[0]]=testY0

dataX=np.zeros([org.shape[0]+testX0.shape[0],testX0.shape[1]])
dataX[0:org.shape[0]]=org
dataX[org.shape[0]:dataX.shape[0]]=testX0

blurX=np.copy(dataX); blurX[0:org.shape[0]]=blur;
flipHX=np.copy(dataX); flipHX[0:org.shape[0]]=flipH;
flipVX=np.copy(dataX); flipVX[0:org.shape[0]]=flipV;
rotate90X=np.copy(dataX); rotate90X[0:org.shape[0]]=rotate90;
rotate180X=np.copy(dataX); rotate180X[0:org.shape[0]]=rotate180;
scaleX=np.copy(dataX); scaleX[0:org.shape[0]]=scale;

model = load_model('res10_L4/model_10k_100_sgd01-dec_m5_L0_l2d')
model.summary()

model0 = load_model('res10_L4/model_10k_100_sgd01_m0_L0_a')
model0.summary()

# evaluate the network
print("[INFO] evaluating network...")

res=np.zeros([8,1])
res[0]=classification_report(testY.argmax(axis=1), model.predict(testX, batch_size=32).argmax(axis=1), target_names=labelNames, output_dict=True)['truck']['precision']
for k, X in enumerate([dataX,scaleX,blurX,flipHX,flipVX,rotate180X,rotate90X]):
	res[k+1]=classification_report(dataY.argmax(axis=1), model.predict(X, batch_size=32).argmax(axis=1), target_names=labelNames, output_dict=True)['truck']['precision']

# evaluate the network
print("[INFO] evaluating network...")

res0=np.zeros([8,1])
res0[0]=classification_report(testY.argmax(axis=1), model0.predict(testX, batch_size=32).argmax(axis=1), target_names=labelNames, output_dict=True)['truck']['precision']
for k, X in enumerate([dataX,scaleX,blurX,flipHX,flipVX,rotate180X,rotate90X]):
	res0[k+1]=classification_report(dataY.argmax(axis=1), model0.predict(X, batch_size=32).argmax(axis=1), target_names=labelNames, output_dict=True)['truck']['precision']


plt.figure()
plt.bar(np.arange(0, len(res)),res0[:,0],width=-0.4,align='edge')
plt.bar(np.arange(0, len(res)),res[:,0],width=0.4,align='edge')
plt.xticks(np.arange(0, len(res)),['cifar-10', 'original', 'scale', 'blur', 'flipH', 'flipV', 'rot180', 'rot90'])
plt.title("Classification Precision for Trucks \n Model: 4L-SGD (blue) vs. 4L-SGD*-M-L2d (orange)");
plt.xlabel("Test Data");
plt.ylabel("Classification Precision");
plt.savefig('output/try.png');


pdb.set_trace() #mz
