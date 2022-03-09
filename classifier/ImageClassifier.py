from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import cv2
import sys, os, shutil
# import tensorflow as tf
from PIL import Image
from tflite_runtime.interpreter import Interpreter
from django.conf import settings

sys.path.append(os.path.dirname(os.getcwd()))
out_path =os.path.join(settings.BASE_DIR, 'media')
outurl = '/media/'

import logging

#################################################
logger=logging.getLogger("Classifier")
c_handler = logging.StreamHandler()

logger.setLevel('INFO')
c_format = logging.Formatter('%(asctime)s %(module)s %(levelname)s %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
##################################################


class Classify():
    def __init__(self, label_path, model_path):
        base_classfier_models=f"{settings.BASE_DIR}{os.sep}classifier{os.sep}models"
        label_path=f"{base_classfier_models}{os.sep}{label_path}"
        model_path=f"{base_classfier_models}{os.sep}{model_path}"
        logger.info("Loading Model")
        self.labels = self.load_labels(label_path)
        self.interpreter = Interpreter(model_path)
        logger.info("Loaded Model")
        self.interpreter.allocate_tensors()
        _, self.height, self.width, _ = self.interpreter.get_input_details()[0]['shape']


    def load_labels(self, path):
        with open(path, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f.readlines())}


    def set_input_tensor(self, image):
        tensor_index = self.interpreter.get_input_details()[0]['index']
        self.interpreter.set_tensor(tensor_index, image)
        # self.input_tensor = self.interpreter.tensor(tensor_index)()[0]
        # self.input_tensor[:, :] = image


    def classify_image(self, image, top_k=1):
        """Returns a sorted array of classification results."""
        self.set_input_tensor(image)
        self.interpreter.invoke()
        output_details = self.interpreter.get_output_details()[0]
        print(output_details)
        output = np.squeeze(self.interpreter.get_tensor(output_details['index']))
        # output = self.interpreter.get_tensor(output_details['index'])
        print(output)
        # If the model is quantized (uint8 data), then dequantize the results
        if output_details['dtype'] == np.uint8:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point)
        print(output)
        # ordered = np.argpartition(-output, top_k)
        # return [(i, output[i]) for i in ordered[:top_k]]
        if output >=0.5:
            return "wheat"
        else:
            return "Other"


    def run(self, image):
        start_time=time.time()
        logger.info("classifying input file")
        # image=cv2.resize(image, (self.width,self.height))
        shutil.copy(image,f"{out_path}{os.sep}")
        image= Image.open(image).convert('RGB').resize((self.width, self.height),
                                                         Image.ANTIALIAS)
        img_array = np.array( image, dtype=np.float32 )
        img_array= np.expand_dims(img_array, axis=0)
        results = self.classify_image(img_array)
        elapsed_ms = (time.time() - start_time) * 1000
        label_id, prob = results, 1
        logger.info("{} detected with with prob {} in {} milliseconds".format(label_id,prob,elapsed_ms))
        return results
