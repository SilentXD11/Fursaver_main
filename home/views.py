from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from django.shortcuts import render

# Create your views here.
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from django.http import HttpResponse

modeldog = load_model('C:/Users/HP/OneDrive/Desktop/FurSaver-Main/modeldisease/modeldisease')
modelcat = load_model('C:/Users/HP/OneDrive/Desktop/FurSaver-Main/modeldisease/modeldisease')
labels=['Bacteria', 'Ringworm']
# sample labels
import numpy as np
from tensorflow.keras.preprocessing import image

# Define the disease labels
disease_labels = ['Bacteria', 'Ringworm']

def pridictdog(path):
    img = image.load_img(path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Make predictions
    predictions = modeldog.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class = predicted_class_index[0]

    # Map the predicted class index to disease label
    predicted_disease = disease_labels[predicted_class]

    return predicted_disease




# Define the disease labels
disease_labels = ['Bacteria', 'Ringworm']

def pridictcat(path):
    img = image.load_img(path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Make predictions
    predictions = modelcat.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class = predicted_class_index[0]

    # Map the predicted class index to disease label
    predicted_disease = disease_labels[predicted_class]

    return predicted_disease


def home(request):
    return render(request, 'home/home.html')

def community(request):
    return render(request, 'home/community.html')    

def sub(request):
    return render(request, 'home/userinfo.html') 

def donation(request):
    return render(request, 'home/donation.html') 


def uploadcat(request):
    if request.method == 'POST':
        print(request)
        print (request)
        print (request.POST.dict())
        fileObj=request.FILES['filePath']

        fs=FileSystemStorage()
        filePathName=fs.save(fileObj.name,fileObj)
        print('+/'*100,filePathName)
        import os
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print('*'*100,BASE_DIR)
        a=pridictcat(BASE_DIR+'/media/'+filePathName)
        print('+'*100,a)
        filePathName=fs.url(filePathName)
        # ["","hair_loss", "patch", "ticks", ""]
        if a == 'Bacteria' :
            result = 'Bacteria'
            info = 'Need to deworm '
        elif a== 'Ringworm' :
            result = 'Ringworm'
            info = 'Dermatitis but medicine is based on weight. Immediate relief with lotion '
        
        

        return render(request, 'home/uploadcat.html',{'result':result,'info':info,'filePathName':filePathName})
    else:
        return render(request, 'home/uploadcat.html')
def uploaddog(request):
    if request.method == 'POST':
        print(request)
        print (request)
        print (request.POST.dict())
        fileObj=request.FILES['filePath']

        fs=FileSystemStorage()
        filePathName=fs.save(fileObj.name,fileObj)
        print('+/'*100,filePathName)
        import os
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print('*'*100,BASE_DIR)
        a=pridictdog(BASE_DIR+'/media/'+filePathName)
        print('+'*100,a)
        filePathName=fs.url(filePathName)
        if a == 'Bacteria' :
            result = 'Bacteria'
            info = 'Need to deworm '
        elif a== 'Ringworm' :
            result = 'Ringworm'
            info = 'Dermatitis but medicine is based on weight. Immediate relief with lotion '
        

        return render(request, 'home/uploaddog.html',{'result':result,'info':info,'filePathName':filePathName})
    else:
        return render(request, 'home/uploaddog.html')

def about(request):
    return render(request, 'home/aboutus.html')

def dogdescription(request):
    return render(request, 'home/dogdescription.html')

def catdescription(request):
    return render(request, 'home/catdescription.html') 



