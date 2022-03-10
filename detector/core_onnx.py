from turtle import right
import cv2
import time
import sys
import numpy as np
import os
from django.conf import settings


class DetectONNX():
    def __init__(self) -> None:
        self.INPUT_WIDTH = 416
        self.INPUT_HEIGHT = 416
        self.SCORE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.5
        self.CONFIDENCE_THRESHOLD = 0.5
        self.is_cuda = True if settings.DEVICE == "cuda" else False
        # self.is_cuda = False
        self.colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        self.start = time.time_ns()
        frame_count = 0
        self.total_frames = 0
        fps = -1
        self.out_path =os.path.join(settings.BASE_DIR, 'media')
        # self.out_path = '../media'
        self.outurl = '/media/'
        self.load_classes()
        self.build_model()

    def build_model(self, model="detector/models/best.onnx"):
        self.net = cv2.dnn.readNet(model)
        if self.is_cuda:
            print("Attempty to use CUDA")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        return preds


    def load_classes(self, classes_file="detector/models/classes.txt"):
        self.class_list = []
        with open(classes_file, "r") as f:
            self.class_list = [cname.strip() for cname in f.readlines()]


    def wrap_detection(self, input_image, output_data):
        class_ids = []
        confidences = []
        boxes = []

        rows = output_data.shape[0]

        image_width, image_height, _ = input_image.shape

        x_factor = image_width / self.INPUT_WIDTH
        y_factor =  image_height / self.INPUT_HEIGHT

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= 0.4:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .25):

                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    left = 0 if left < 0 else left
                    top = 0 if top < 0 else top
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_class_ids, result_confidences, result_boxes

    def format_yolov5(self, frame):

        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def run_detection(self, path):

        file_name = path.split("/")[-1]
        frame=cv2.imread(path)
        classify_frame= frame.copy()
        if frame is None:
            print("End of stream")
            

        inputImage = self.format_yolov5(frame)
        # inputImage = frame

        outs = self.detect(inputImage)
        # outs = detect(inputImage, net)

        class_ids, confidences, boxes = self.wrap_detection(inputImage, outs[0])
        number_of_grains = 0
        # frame_count += 1
        # total_frames += 1
        data = {}
        i=0
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = self.colors[int(classid) % len(self.colors)]
            x, y, w, h = box
            print(x, y, w, h)
            number_of_grains += 1
            label = self.class_list[classid]
            if label in data.keys():
                data[label]['count']+=1
                data[label]['annots'].extend([x,y,w,h])
            else:
                data[label]={}
                data[label]['count']=1
                data[label]['annots']=[[x,y,w,h]]

            self.store_for_classification(file_name, classify_frame, box, i, self.class_list[classid])
            i += 1

            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, self.class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
            

        cv2.imwrite(f"{self.out_path}{os.sep}in_{file_name}",inputImage)
        cv2.imwrite(f"{self.out_path}{os.sep}out_{file_name}",frame)
        out_url = f"{self.outurl}out_{file_name}"
        in_url = f"{self.outurl}in_{file_name}"
        # print(data, out_url, number_of_grains, in_url)
        return data, out_url, number_of_grains, in_url

    def store_for_classification(self, file_name, frame, box, id, class_name):
        x,y,w,h = box
        #for classification
        classify_dir = file_name.split(".")[0]
        classify_dir = f"{self.out_path}{os.sep}classify_{classify_dir}{os.sep}{class_name}"
        print(f"storing in dir {classify_dir}")
        if not os.path.exists(f"{classify_dir}"):
            os.makedirs(f"{classify_dir}")

        cv2.imwrite(f"{classify_dir}{os.sep}{id}.jpg",frame[y:y+h, x:x+w])
        

        
# print("Total frames: " + str(total_frames))
# run_detection('/Volumes/Projects/Projects/cv/chana/train/images/broken-11_jpg.rf.ef1b5accc6cb445b5cc534f03353e0cd.jpg')
# import glob
# ut=DetectONNX()
# base_dir= "/Volumes/Projects/Projects/cv/chana/"
# for file in glob.glob(f'{base_dir}/*/images/*.jpg'):
#     ut.run_detection(file)