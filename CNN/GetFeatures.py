import os
import cv2
import numpy as np
import tensorflow as tf

dataPath='test'
#dataPath='testSmall'
CHARS = ["-","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

def readData(path):
	images=[]
	plates=[]
	for subdir,dir,files in os.walk(path):
		for file in files:
			plates.append(file)
			img=cv2.imread(path+'/'+file)
			img=img.flatten()
			images.append(img)
	return images,plates

def formatPlates(plates):
	formattedPlates=[]
	for plate in plates:
		plateArr=[]
		plate=plate.replace(" ","").replace("\n","")[:-4]
		for ch in plate:
			tmp=np.zeros(len(CHARS))
			tmp[CHARS.index(ch)]=1
			plateArr=plateArr+tmp.tolist()
		while(len(plateArr)<(7*len(CHARS))):
			plateArr=plateArr+np.zeros(len(CHARS)).tolist()
		formattedPlates.append(plateArr)
	return formattedPlates

def getTrainandTestData(images,plates,testSize=0.6):
	images=np.array(images)
	plates=np.array(plates)
	testingSize=int(testSize*len(images))

	train_x=list(images[:-testingSize])
	train_y=list(plates[:-testingSize])
	test_x=list(images[-testingSize:])
	test_y=list(plates[-testingSize:])

	return train_x,train_y,test_x,test_y

def getTestData(images,plates):
	images=np.array(images)
	plates=np.array(plates)

	test_x=list(images[-200:])
	test_y=list(plates[-200:])

	return test_x,test_y

if __name__=="__main__":
	images,plates=readData(dataPath)
	plates=formatPlates(plates)

	train_x,train_y,test_x,test_y=getTrainandTestData(images,plates)
	
	print test_y
	print train_y
