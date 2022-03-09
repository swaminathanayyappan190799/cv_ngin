
# from ctypes import *
# import math
# import random, sys, cv2
# from PIL import Image
# import threading, time
# import the necessary packages
import numpy as np
import argparse
import time
import datetime
import cv2
import os, re
import logging
from threading import Thread
from multiprocessing import Process, cpu_count

# from logsNError.core import initialize_logging, LogDBHandler
# from videoProcessor import np_cv

#################################################
logger=logging.getLogger("Detector")
c_handler = logging.StreamHandler()

logger.setLevel('INFO')
c_format = logging.Formatter('%(asctime)s %(module)s %(levelname)s %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
##################################################

import configparser
from django.conf import settings

# load the character class labels our YOLO model was trained on
labelfile="wheat"
labelsPath = f"{settings.BASE_DIR}{os.sep}detector{os.sep}models{os.sep}{labelfile}_labels.txt"
CHAR_LABELS = open(labelsPath).read().strip().split("\n")
 
# initialize a list of colors to represent each possible char label
np.random.seed(42)
CHAR_COLORS = np.random.randint(0, 255, size=(len(CHAR_LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration for numberplates

# derive the paths to the YOLO weights and model configuration for characters

weights=f"{settings.BASE_DIR}{os.sep}detector{os.sep}models{os.sep}yolov3-tiny_best_wheat.weights"
config=f"{settings.BASE_DIR}{os.sep}detector{os.sep}models{os.sep}yolov3-tiny.cfg"
 
# load our YOLO object detector trained on numberplate dataset (1 classes)
logger.info("Loading YOLO from disk...")
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our YOLO object detector trained on character dataset (36 classes)
char_net = cv2.dnn.readNetFromDarknet(config, weights)

reqd_confidence = 0.1
threshold = 0.1
#placement of window
out_path =os.path.join(settings.BASE_DIR, 'media')
outurl = '/media/'

showVideo = False
def show_img(label, image):
	try:
		if showVideo == "True":
			# print("showing image")
			cv2.imshow(label,image)
			cv2.waitKey(10)
	except:
		pass


class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
 
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
 
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
 
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
 
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
 
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()
 
def processVideosFrompath(path):	
	imagelist=[]
	if os.path.isfile(path):
				# print("Reading file "+path+"/"+img_file)
				img=cv2.imread(path)
				imagelist.append({'name':path.split('/')[-1],'image':img})
				# imagelist.append(img)
				# print("Processing filelist "+path)
				start = time.time()
				# p = Thread(target=getNeededFrame, args=(imagelist,'file'))	
				data, image, number_of_grains, input=getNeededcharFromFrame(imagelist,'file')
				end = time.time()
				logger.info("YOLO took {:.6f} seconds for this frame ".format(end - start))
				os.remove(path)
	else:
		for img_file in sorted(os.listdir(path)):
			imagelist=[]
			if img_file.lower().find('.jp') > -1:
				# print("Reading file "+path+"/"+img_file)
				img=cv2.imread(path+"/"+img_file)
				imagelist.append({'name':img_file,'image':img})
				# imagelist.append(img)
				# print("Processing filelist "+path+"/"+img_file)
				start = time.time()
				# p = Thread(target=getNeededFrame, args=(imagelist,'file'))	
				data, image, number_of_grains, input, count_by_type=getNeededcharFromFrame(imagelist,'file')
				end = time.time()
				logger.info("YOLO took {:.6f} seconds for this frame ".format(end - start))	
	return data, image, number_of_grains, input


def getNeededcharFromFrame(imagelist,camid):
    try:
		# # print(os.listdir(path))
		# imagelist=os.listdir(imagelist)
		# imagelist.sort()
        x,y,h,w=(0,0,0,0)
        for obj in imagelist:
        #load image
            # print("[INFO] Processing file {}".format(camid))
        # image=cv2.imread(path+imagefile)
            image=obj['image'].copy()
            found=False #used to save totrainingpath
            data ={}
            annots=[]
            # print("file: ",obj['name'])
            # load our input image and grab its spatial dimensions
            # image = array_to_image(img)
            (H, W, C) = image.shape

                
            # determine only the *output* layer names that we need from YOLO
            ln = char_net.getLayerNames()
            ln = [ln[i - 1] for i in char_net.getUnconnectedOutLayers()]
            # construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
            char_net.setInput(blob)
            start = time.time()
            layerOutputs = char_net.forward(ln)
            end = time.time()

            # show timing information on YOLO
            logger.debug("YOLO took {:.6f} seconds for frame ".format(end - start))
            # fps.update()


            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                #arun added
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                
                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability

                    if confidence > reqd_confidence:
                        # print(confidence)
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                
                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        
                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, reqd_confidence, threshold)
            number_of_grains=len(idxs)

            # ensure at least one detection exists
            # numplate=[]
            # data=obj['name']
            # ordered_np={}
            # char_img=np.empty(2)
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in np.sort(idxs.flatten()):
                    # print(i)
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    label=CHAR_LABELS[classIDs[i]] 
                    if label in data.keys():
                        data[label]['count']+=1
                        data[label]['annots'].extend([x,y,x+w,y+h])
                    else:
                        data[label]={}
                        data[label]['count']=1
                        data[label]['annots']=[[x,y,x+w,y+h]]
                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in CHAR_COLORS[classIDs[i]]]
                    image=cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.1f}".format(CHAR_LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        2, color, 3)
                    # show_img('output',cv2.resize(char_img,(640,480)))
                    # ordered_np[x]= CHAR_LABELS[classIDs[i]] 
            logger.info(f"saving file {out_path}{os.sep}{obj['name']}")
            cv2.imwrite(f"{out_path}{os.sep}in_{obj['name']}",obj['image'])
            cv2.imwrite(f"{out_path}{os.sep}out_{obj['name']}",image)
            out_url = f"{outurl}out_{obj['name']}"
            in_url = f"{outurl}in_{obj['name']}"
            print(data, out_url, number_of_grains, in_url)
            return data, out_url, number_of_grains, in_url
        # return data
    except Exception as e:
        # print(str(e))
        logger.error(str(e),exc_info=True)
        return False