
# Importing essential libraries and modules

from flask import Flask, render_template, request , redirect, url_for
import numpy as np
import pandas as pd
import requests
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.RNet_model import PlantDiseaseAndCategoryClassification
from utils.config import disease_classes, plant_category
# 



## loading and setting model
disease_model_path = 'Models/plant-diseaseandcategory-model.pth'
disease_model = PlantDiseaseAndCategoryClassification(3, len(disease_classes), len(plant_category))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


app = Flask(__name__)

@app.route("/", methods = ['GET'])
def home():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    title = "Crops Disease and Category Identification"
    if request.method =="POST": 
        if "file" not in request.files:
            return redirect(redirect.url)
        file = request.files.get("file")
        if not file:
            return redirect(url_for('home'))
        try : 
            img = file.read()
            prediction = predict_result(img)
            #prediction = Markup(str())
            #print(prediction)
            return render_template('prediction.html', status = 200, result = prediction, title=title)

        except Exception as exp:
            print(exp) 
             
    return render_template("index.html", title=title)  # Redirect to the home page
    # return render_template('index.html', status=500, res = "Internal Server Error ")

def predict_result(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    leaf_image = Image.open(io.BytesIO(img))
    leaf_image_t = transform(leaf_image)
    leaf_image_u = torch.unsqueeze(leaf_image_t, 0)

    # get prediction
    disease, category = model(leaf_image_u)
    _,p_disease = torch.max(disease, dim=1)
    _, p_category = torch.max(category, dim=1)

    #print(p_disease)
    #print(p_category)
    predicted_disease = disease_classes[p_disease[0].item()]
    predicted_category = plant_category[p_category[0].item()]

    return (predicted_disease, predicted_category)

if __name__ == "__main__":
    app.run(debug =True)