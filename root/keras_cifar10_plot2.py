# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pdb #mz pdb.set_trace() #mz
from imutils import paths
import os
N_EPOCH=100

#Loading:
file_paths = list(paths.list_files('res10_L3',validExts="npy"))
hist3=[]
for fpath in file_paths:
	h=np.load(fpath,allow_pickle='TRUE').item()
	h['name']=fpath.split(os.path.sep)[-1]
	hist3.append(h)

file_paths = list(paths.list_files('res10_L4',validExts="npy"))
hist4=[]
for fpath in file_paths:
	h=np.load(fpath,allow_pickle='TRUE').item()
	h['name']=fpath.split(os.path.sep)[-1]
	hist4.append(h)
pdb.set_trace() #mz
names3=['3L-SGD*-M-L2','SGD*','3L-SGD*-M-L2d','SGD*-L2d', 'SGD-M-L2d','SGD-M-L2','SGD','SGD-L2', 'SGD*-L2','3L-SGD*-M','SGD-M','SGD-L2d']

names4=['4L-SGD*-M-L2','4L-SGD*-M-L2d','4L-SGD', '4L-SGD*-M','4L-SGD*-M-L2'] 
# no b 2 3 0 1
hists=[hist3[9],hist3[0],hist3[2],hist4[2],hist4[3],hist4[0],hist4[1]]

names=[names3[9],names3[0],names3[2],names4[2],names4[3],names4[0],names4[1]]


res=np.zeros([len(hists),5])
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
markers=['^','^','^','d','d','d','d']
colors=['y','y','y','k','k','k','k']

plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, len(hists)), res[:,3], label="tr_loss")
plt.plot(np.arange(0, len(hists)), res[:,4], label="val_loss")


for k in np.arange(0, len(hists)):
	plt.scatter(k, res[k,3], c=colors[k], s=70, marker=markers[k])
	plt.scatter(k, res[k,4], c=colors[k], s=70, marker=markers[k])

plt.xticks(np.arange(0, len(hists)),names,rotation=20)
plt.title("Training & Validation Loss \n 3-Layer [3072x1024x10] vs. 4-Layer [3072x1024x512x10]")
plt.xlabel("Model Specification")
plt.ylabel("Loss")
plt.legend()
plt.savefig('output/3Lvs4L.png') #plt.savefig(args["output"])

pdb.set_trace() #mz

		#plt.plot(np.arange(0, N_EPOCH), history["loss"], label="train_loss")
		#plt.plot(np.arange(0, N_EPOCH), history["val_loss"], '--', label="val_loss")
		#plt.plot(np.arange(0, N_EPOCH), history["accuracy"], '-.', label="train_acc")
		#plt.plot(np.arange(0, N_EPOCH), history["val_accuracy"], '-', label=history['name'].split('_')[4:-1]) #"val_acc"

#pdb.set_trace() #mz
#plt.plot(np.arange(0, len(hist)), res[:,0], label="train_loss")
#plt.plot(np.arange(0, len(hist)), res[:,1], label="val_loss")
#plt.plot(np.arange(0, len(hist)), res[:,2], label="train_acc")
