from cProfile import run
import cv2
import numpy
from django.shortcuts import render
from django.template import RequestContext
from django.http import HttpResponseRedirect



from api.forms import FileInputForm

from classifier.ImageClassifier import Classify
from detector.core_onnx import DetectONNX

pipeline = [Classify('tf_labels.txt','256_grain_model.tflite'), {'wheat':""}]

# def upload(request):
#     # Handle file upload
    
#     if request.method == 'POST':
#         input_file=request.FILES.get('inputfile')
#         with open(f"/tmp/{input_file.name}","wb") as fh:
#             for chunk in input_file.chunks():
#                 fh.write(chunk)


#         # input_file=cv2.imread(f"/tmp/{input_file.name}")
#         print("Stating Classification")
#         # img_classifier = Classify('tf_labels.txt','320_monkey_model.tflite')
#         grain=pipeline[0].run(f"/tmp/{input_file.name}")
#         if grain == 'wheat':
#             # pipeline[1]['wheat'].run(f"/tmp/{input_file.name}")
#             data, url, number_of_grains, input = processVideosFrompath(f"/tmp/{input_file.name}")
#         else:
#             data ={}
#             url = f"/media/{input_file.name}"
#             input = f"/media/{input_file.name}"
#             count_by_type={}
#             number_of_grains = 0
#         # print(data)
#         form = FileInputForm()
#         context = {"form": form,
#                     "output":{"grain":grain, "data": data, "number":number_of_grains, "url":url, "input":input}}
#     else:
#         form = FileInputForm()
#         context = {'form': form} # A empty, unbound form

#     # Load documents for the list page
    
#     # Render list page with the documents and the form
#     return render(
#         request,
#         'api/templates/upload.html',
#         context,        
#     )

def detect_chana(request):
    # Handle file upload
    
    if request.method == 'POST':
        input_file=request.FILES.get('inputfile')
        with open(f"/tmp/{input_file.name}","wb") as fh:
            for chunk in input_file.chunks():
                fh.write(chunk)

        print("Starting Chana Detection")
        detection = DetectONNX()
        data, url, number_of_grains, input = detection.run_detection(f"/tmp/{input_file.name}")
        # print(data)
        form = FileInputForm()
        context = {"form": form,
                    "output":{"grain":"chana", "data": data, "number":number_of_grains, "url":url, "input":input}}
    else:
        form = FileInputForm()
        context = {'form': form} # A empty, unbound form

    # Load documents for the list page
    
    # Render list page with the documents and the form
    return render(
        request,
        'api/templates/upload.html',
        context,        
    )