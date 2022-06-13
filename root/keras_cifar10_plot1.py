# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pdb #mz pdb.set_trace() #mz
from imutils import paths
import os
N_EPOCH=100

#Loading:
file_paths = list(paths.list_files('res10_L3',validExts="npy"))
hist=[]
for fpath in file_paths:
	h=np.load(fpath,allow_pickle='TRUE').item()
	h['name']=fpath.split(os.path.sep)[-1]
	hist.append(h)


names=['SGD*-M-L2','SGD*','SGD*-M-L2d','SGD*-L2d', 'SGD-M-L2d','SGD-M-L2','SGD','SGD-L2', 'SGD*-L2','SGD*-M','SGD-M','SGD-L2d']
hists=[hist[6],hist[7],hist[11],hist[1],hist[8],hist[3],hist[10],hist[5],hist[4],
hist[9],hist[0],hist[2]]#,hist4[0],hist4[1],hist4[2]]

names=[names[6],names[7],names[11],names[1],names[8],names[3],names[10],names[5],names[4],
names[9],names[0],names[2]]

res=np.zeros([len(hist),5])
for i,history in enumerate(hists):
		#pdb.set_trace() #mz
	res[i,0]=sum(history["loss"][80:100])/20
	res[i,1]=sum(history["val_loss"][80:100])/20
	res[i,2]=sum(history["accuracy"][80:100])/20
	res[i,3]=sum(history["val_accuracy"][80:100])/20
	res[i,4]=max(history["val_accuracy"][80:100])
	#names.append(history['name'].split('_')[4:-1]) #"val_acc")
	print("number= {} name= {}".format(i,history['name']))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()

markers=['o','o','o','d','d','d','p','p','p','^','^','^']
colors=['g','g','g','m','m','m','c','c','c','y','y','y']

plt.plot(np.arange(0, len(hist)), res[:,3], label="tr_loss")
plt.plot(np.arange(0, len(hist)), res[:,4], label="val_loss")


for k in np.arange(0, len(hist)):
	plt.scatter(k, res[k,3], c=colors[k], s=70, marker=markers[k])
	plt.scatter(k, res[k,4], c=colors[k], s=70, marker=markers[k])

plt.xticks(np.arange(0, len(hist)),names,rotation=20)
plt.title("Training & Validation Loss \n 3-Layer NN [3072 x 1024 x 10]")
plt.xlabel("Model Specification")
plt.ylabel("Loss")
plt.legend()
plt.savefig('output/3L.png') #plt.savefig(args["output"])

pdb.set_trace() #mz

		#plt.plot(np.arange(0, N_EPOCH), history["loss"], label="train_loss")
		#plt.plot(np.arange(0, N_EPOCH), history["val_loss"], '--', label="val_loss")
		#plt.plot(np.arange(0, N_EPOCH), history["accuracy"], '-.', label="train_acc")
		#plt.plot(np.arange(0, N_EPOCH), history["val_accuracy"], '-', label=history['name'].split('_')[4:-1]) #"val_acc"

#pdb.set_trace() #mz
#plt.plot(np.arange(0, len(hist)), res[:,0], label="train_loss")
#plt.plot(np.arange(0, len(hist)), res[:,1], label="val_loss")
#plt.plot(np.arange(0, len(hist)), res[:,2], label="train_acc")
