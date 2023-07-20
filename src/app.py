
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