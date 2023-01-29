import base64
import enum
import imp
import io
import json
import os
import shutil
from zipfile import ZipFile
from matplotlib.pyplot import annotate

from torchvision import models
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from django.http.response import HttpResponse

from django.core.files.storage import FileSystemStorage
from .forms import FileUploadForm, ImageUploadForm,TxtUploadForm, CorrectionZipUploadForm

import mimetypes
import zipfile
import string
import argparse

from datetime import datetime
import pytz

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

# from .utils import CTCLabelConverter, AttnLabelConverter
# from .dataset import RawDataset, AlignCollate
# from .model import Model
from .detect_lines import detect_lines
from .detect_boxes import detect_boxes

# PyTorch-related code from: https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
# load pretrained DenseNet and go straight to evaluation mode for inference
# load as global variable here, to avoid expensive reloads with each request
# model = models.densenet121(pretrained=True)
# model.eval()

# load mapping of ImageNet index to human-readable label
# run "python manage.py collectstatic" first!
# json_path = os.path.join(settings.STATICFILES_DIRS[0], "imagenet_class_index.json")
# json_path = os.path.join(settings.STATIC_ROOT, "imagenet_class_index.json")
# imagenet_mapping = json.load(open(json_path))


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# .convert('L')
# image = image.transpose(Image.FLIP_LEFT_RIGHT)
# width, height = image.size
# print (width, height)


def get_prediction_from_image():
    """
    Transform image into required DenseNet format: 224x224 with 3 RGB channels and normalized.
    Return the corresponding tensor.
    """
    
    #TODO :- Crop the Page and Store the Text Lines in another folder.
    dirpath = "vis"
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    
    print("Detecting lines...")
    detect_boxes()
    print("Completed!")

    # Detect Lines in the Image.
    opt = Namespace(FeatureExtraction='UNet', PAD=False, Prediction='CTC', SequenceModeling='DBiLSTM', Transformation='None', batch_max_length=100, batch_size=1, hidden_size=256, image_folder='vis', imgH=32, imgW=400, input_channel=1, num_fiducial=20, num_gpu=1, output_channel=512, rgb=False, saved_model='transcriber/best_norm_ED.pth', sensitive=False, workers=4)
    file = open("transcriber/UrduGlyphs.txt","r",encoding="utf-8")
    content = file.readlines()
    content = ''.join([str(elem).strip('\n') for elem in content])
    opt.character = content+" "   #+"abcdefghijklmnopqrstuvwxyz"+"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    file.close()
    print("Reading lines...")
    final_text = detect_lines(opt)
    print("Completed!")
    
    # demo(opt)
    return final_text

def save_correction(file_object):
    now = datetime.now(pytz.timezone('asia/kolkata'))
    if  not os.path.exists("corrections"):
        os.mkdir("corrections")
    annotation_dir = os.path.join("corrections",now.strftime("%d_%m_%Y_%H_%M_%S"))
    if os.path.exists(annotation_dir):
        shutil.rmtree(annotation_dir)
    os.mkdir(annotation_dir)
    with open(os.path.join(annotation_dir,"gt.txt"), 'wb+') as f:
        for chunk in file_object.chunks():
            f.write(chunk)
    if os.path.exists("vis"):
        line_images = sorted(os.listdir("vis/"))
        for i,line in enumerate(line_images):
            if str(line).endswith(".jpg"):
                shutil.move(os.path.join("vis",line),annotation_dir+"/"+str(i).zfill(3)+".jpg")
        shutil.rmtree("vis")

    lines = ""
    with open(os.path.join(annotation_dir,"gt.txt"),"r",encoding="utf-8") as gtFile:
        old_lines = gtFile.readlines()
        for i,line in enumerate(old_lines):
            line = str(i).zfill(3)+".jpg" + "\t" + line.strip("\n")
            lines = lines + line + "\n"
    with open(os.path.join(annotation_dir,"gt.txt"),"w",encoding="utf-8") as gtFile:
        gtFile.write(lines)

def is_image(file):
    return (file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG"))

def save_each_in_zip(pred,name):
    shutil.move("static/vis_test.jpg",os.path.join("predicted_zip",name) + ".jpg")
    with open(os.path.join("predicted_zip",name) + ".txt","w",encoding="utf-8") as GT:
        GT.write(pred)
    os.mkdir(os.path.join("predicted_zip",name))
    for item in os.listdir("vis"):
        new_item = "cropped_" + "".join(item.split("_test"))
        if item.endswith(".jpg"):
            shutil.move(os.path.join("vis",item),os.path.join("predicted_zip",name,new_item))

def annotate_extracted_zip():
    arr = []
    for i,lis in enumerate(os.listdir("extracted_zip")):
        imgFile = os.path.join("extracted_zip",lis)
        image = Image.open(imgFile)
        image = image.convert('RGB')
        image.save ("img/test.jpg")
        dirpath = "vis"
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        print("Detecting lines...")
        detect_boxes()
        print("Completed!")
        opt = Namespace(FeatureExtraction='UNet', PAD=False, Prediction='CTC', SequenceModeling='DBiLSTM', Transformation='None', batch_max_length=100, batch_size=1, hidden_size=256, image_folder='vis', imgH=32, imgW=400, input_channel=1, num_fiducial=20, num_gpu=1, output_channel=512, rgb=False, saved_model='transcriber/best_norm_ED.pth', sensitive=False, workers=4)
        file = open("transcriber/UrduGlyphs.txt","r",encoding="utf-8")
        content = file.readlines()
        content = ''.join([str(elem).strip('\n') for elem in content])
        opt.character = content+" "   #+"abcdefghijklmnopqrstuvwxyz"+"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        file.close()
        print("Reading lines...")
        final_text = detect_lines(opt)
        print("Completed!")
        sub_dir_name = str(i)
        if os.path.exists("vis/predictions.pth"):
            os.remove("vis/predictions.pth")
        save_each_in_zip(final_text,sub_dir_name)
        arr.append(["vis/"+str(i)+".jpg",final_text])
    shutil.make_archive("predictions", 'zip', "predicted_zip")
    return arr

def move_img_with_lines_to_vis(n):
    for i in range(n):
        shutil.copy(os.path.join("predicted_zip",str(i)+".jpg"),os.path.join("static","vis",str(i)+".jpg"))

def annotate_zip(zipFIle):
    # Correctly save all files & directories first
    if os.path.exists("extracted_zip"):
        shutil.rmtree("extracted_zip")
    os.mkdir("extracted_zip")
    if os.path.exists("predicted_zip"):
        shutil.rmtree("predicted_zip")
    os.mkdir("predicted_zip")
    with zipfile.ZipFile(zipFIle, 'r') as zip_ref:
        zip_ref.extractall("extracted_zip")
    images = []
    counter = 0
    files = os.listdir("extracted_zip")
    for subdir, dirs, files in os.walk("extracted_zip"):
        for file in files:
            each_file = os.path.join(subdir, file)
            if each_file.endswith('.jpg') or each_file.endswith('.jpeg') or each_file.endswith('.png') or each_file.endswith('.JPG') or each_file.endswith('.JPEG') or each_file.endswith('.PNG'):
                counter+=1
                if counter>=10:
                    break
                images.append(each_file)
        if counter>=10:
            break
    if os.path.exists("static/vis"):
        shutil.rmtree("static/vis")
    os.mkdir("static/vis")
    for i,img in enumerate(images):
        shutil.copy(img,os.path.join("static","vis",str(i))+".jpg")
        shutil.move(img,os.path.join("extracted_zip",str(i))+".jpg")
    for lis in os.listdir("extracted_zip"):
        if os.path.isdir(os.path.join("extracted_zip",lis)):
            shutil.rmtree(os.path.join("extracted_zip",lis))
    arr = annotate_extracted_zip()
    move_img_with_lines_to_vis(len(arr))
    return arr

def save_batch_correction(zipFile):
    now = datetime.now(pytz.timezone('asia/kolkata'))
    if  not os.path.exists("batch_corrections"):
        os.mkdir("batch_corrections")
    annotation_dir = os.path.join("batch_corrections",now.strftime("%d_%m_%Y_%H_%M_%S"))
    if os.path.exists(annotation_dir):
        shutil.rmtree(annotation_dir)
    with zipfile.ZipFile(zipFile, 'r') as zip_ref:
        zip_ref.extractall(annotation_dir)

def index(request):
    corrected = None
    predicted_label = None
    is_image_not_zip = None
    form = None
    form2 = None
    arr = None

    if request.method=='POST':
        form = CorrectionZipUploadForm(request.POST, request.FILES)
        file_object = request.FILES['file']
        if form.is_valid():
            try:
                save_batch_correction(file_object)
                corrected = 1
            except:
                corrected = 0
        else:
            form = TxtUploadForm(request.POST, request.FILES)
            file_object = request.FILES['file']
            if form.is_valid():
                try:
                    save_correction(file_object)
                    corrected = 1
                except:
                    corrected = 0
            else:
                # If uploaded for transcription
                form = FileUploadForm(request.POST, request.FILES)
                if form.is_valid():
                    print("Upload succesfull")
                    file_object = request.FILES['file']
                    file = str(file_object)
                    if is_image(file):
                        is_image_not_zip = True
                        img = Image.open(file_object).convert('RGB')
                        img.save("img/test.jpg")
                        predicted_label = get_prediction_from_image()
                        if os.path.exists("img/test.jpg") and os.path.isfile("img/test.jpg"):
                            # print("Raw Image Deleted")
                            os.remove("img/test.jpg")
                        form2 = TxtUploadForm()
                        form = None
                    else:
                        is_image_not_zip = False
                        arr = annotate_zip(file_object)
                        predicted_label = "~~~" # The characteer ~ is NOT in Urdu Glyphs
                        form2 = CorrectionZipUploadForm()
                        form = None

    else:
        form = FileUploadForm()

    context = {
        'form': form,
        'form2': form2,
        'is_image_not_zip' : is_image_not_zip,
        'predicted_label': predicted_label,
        'Corrected_annotation':corrected,
        'all_labels': arr
    }
    
    return render(request, 'transcriber/index.html', context)

def download_file (request):
    # Define Django Base Directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Define text file name
    filename = "transcription.txt"
    # Define full file path
    filepath = BASE_DIR + '/transcriber/' + filename
    # Open the file for reading content
    path = open(filepath, 'r')
    # Set the mime type
    mime_type, _ = mimetypes.guess_type(filepath)
    # Set the return value of the HttpResponse
    response = HttpResponse(path, content_type=mime_type)
    # Set the HTTP header for sending to browser
    response['Content-Disposition'] = "attachment; filename=%s" % filename
    # Return the response value
    return response

def download_zip (request):
    # Define Django Base Directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Define text file name
    filename = "predictions.zip"
    # Define full file path
    filepath = BASE_DIR + '/' + filename
    # Open the file for reading content
    zip_file = open(filepath, 'rb')
    # Set the return value of the HttpResponse
    response = HttpResponse(zip_file, content_type='application/zip')
    # Set the HTTP header for sending to browser
    response['Content-Disposition'] = "attachment; filename=%s" % filename
    response['Content-Length'] = os.path.getsize(filepath)
    zip_file.close()
    # Return the response value
    return response