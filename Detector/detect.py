#store unsynched plates and times in a file periodically check for connection in another script and push updates once connected
from scipy.misc import imread
import tensorflow as tf
import numpy as np
import thread
import sys
import dbOperations as db
sys.path.insert(0,'../CNN')
import GetFeatures as gf
import cv2
import redis
import time
import csv

CHARS = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

n_classes = 7*len(CHARS)
dropout=0.75
inputData = tf.placeholder('float',[None,None])

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def conv_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,48])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,48,64])),
               'W_conv3':tf.Variable(tf.random_normal([5,5,64,128])),
               'W_fc1':tf.Variable(tf.random_normal([10*20*128,1024])),
               'W_fc2':tf.Variable(tf.random_normal([1024,2048])),
               'out':tf.Variable(tf.random_normal([2048,n_classes])),}

    biases = {'b_conv1':tf.Variable(tf.random_normal([48])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_conv3':tf.Variable(tf.random_normal([128])),
              'b_fc1':tf.Variable(tf.random_normal([1024])),
              'b_fc2':tf.Variable(tf.random_normal([2048])),
              'out':tf.Variable(tf.random_normal([n_classes])),}

    x=tf.reshape(x,shape=[-1,80,160,3])

    conv1=tf.nn.relu(conv2d(x,weights['W_conv1'])+biases['b_conv1'])
    conv1=maxpool2d(conv1)

    conv2=tf.nn.relu(conv2d(conv1,weights['W_conv2'])+biases['b_conv2'])
    conv2=maxpool2d(conv2)

    conv3=tf.nn.relu(conv2d(conv2,weights['W_conv3'])+biases['b_conv3'])
    conv3=maxpool2d(conv3)

    fc1=tf.reshape(conv3,[-1,10*20*128])
    fc1=tf.nn.relu(tf.matmul(fc1,weights['W_fc1'])+biases['b_fc1'])

    fc2=tf.nn.relu(tf.matmul(fc1,weights['W_fc2'])+biases['b_fc2'])
    fc2=tf.nn.dropout(fc2,dropout)

    output=tf.matmul(fc2,weights['out'])+biases['out']

    return output

def normalizeData(codes,sections):
	normalizedCodes=[]
	normalizedSects=[]
	for j,p in enumerate(codes):
  		if p not in normalizedCodes:
			normalizedCodes.append(p)
			normalizedSects.append(sections[j])
	return normalizedCodes,normalizedSects

def scan(image):
	stepSize=40
	windowSize=(160,80)

	images=[]
	sections=[]
	#Chop image into segments
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			section=image[y:y + windowSize[1], x:x + windowSize[0]]
			if(section.shape[1]==160 and section.shape[0]==80):
				img=section.flatten()
				images.append(img)
				sections.append(([y,y + windowSize[1], x,x + windowSize[0]]))

	#Evaluate the value of each segment
	solutions=res.eval({inputData:images},session=sess)
	counter=0
	codes=[]
	for i,sol in enumerate(solutions):
		plate=[]
		for ele in sol:
			plate.append(CHARS[ele])
		codes.append(plate)
		counter+=1
		
	return normalizeData(codes,sections)

#Takes a list of plates and checks if in db and updates detected list
def updateAndAlert(image,plates,sects,r,blacklist):
	with open("detected.csv","a") as det:
		for i,plate in enumerate(plates):
			code=''.join(plate)
			if code in blacklist:
				print("Alert: "+code)
				cv2.rectangle(image, (sects[i][2], sects[i][0]), (sects[i][3], sects[i][1]), (0, 255, 0), 2)
				cv2.imshow("Detector", image)
				cv2.waitKey(1)#Change to 0 for cli img and 1 for camera feed
			det.write(code+","+time.strftime("%c").replace(" ","-")+"\n")

def getImage(cam):
	retval,im=cam.read()
	return im

def shutter(cam):
	temp=getImage(cam)
	return getImage(cam)

def getBlacklist(r):
	return r.smembers("blacklist")

redisDb=db.setupDB() #r=redisDb
cam=cv2.VideoCapture(0)
prediction = conv_neural_network(inputData)

with tf.Session() as sess:
	#Setup Trained model
	saver=tf.train.import_meta_graph('model.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./'))
	sess.run(tf.global_variables_initializer())

	prediction=tf.reshape(prediction,shape=[-1,7,len(CHARS)])
	res = tf.argmax(prediction, 2)

	#Download blacklist
	print("Waiting for connection")
	while True:
		if(db.isConnected()):
			blacklist=getBlacklist(redisDb)
			break
	print("Connected")

	#Thread to scan for connection every ten seconds and update db
	updateDb=thread.start_new_thread(db.update,(redisDb,))

	print("Scanning")
	#Code for single input image from cli
	while True:
		image=imread(sys.argv[1])
		plates,sects=scan(image)
		updateAndAlert(image,plates,sects,redisDb,blacklist)
	'''
	try:
		while(True):
			capture=shutter(cam)
			plates,sects=scan(capture)
			updateAndAlert(capture,plates,sects,redisDb,blacklist)
	except:
		print("Exiting System")'''
	del(cam)