
# Importing essential libraries and modules

from flask import Flask, render_template, request , redirect, url_for
import numpy as np
import pandas as pd
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.RNet_model import ResNet9
# 

# classes of disease
disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']


## loading and setting model
disease_model_path = 'Models/plant_disease_model.pth'
disease_model = ResNet9(3, len(38))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


app = Flask(__name__)

@app.route("/", methods = ['GET'])
def home():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method =="POST": 
        try : 

            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
            ])

            image = Image.open(io.BytesIO(request.files['file']))
            img_t = transform(image)
            img_u = torch.unsqueeze(img_t, 0)

            prediction = None # Some Prediction should be made here 
            print(prediction)

            # Get predictions from model
            yb = disease_model(img_u)
            # Pick index with highest probability
            _, preds = torch.max(yb, dim=1)
            prediction = disease_classes[preds[0].item()]

            return render_template('display.html', status = 200, result = prediction)

        except: 
            pass 
    return redirect(url_for('home'))  # Redirect to the home page
    # return render_template('index.html', status=500, res = "Internal Server Error ")

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    
    

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

if __name__ == "__main__":
    app.run(debug =True)