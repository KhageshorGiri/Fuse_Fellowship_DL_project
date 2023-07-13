
from flask import Flask, render_template, request , redirect, url_for
# 

app = Flask(__name__)

@app.route("/", methods = ['GET'])
def home():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method =="POST": 
        try : 
            file  = request.files['file']
            prediction = None # Some Prediction should be made here 
            print(prediction)
            res = "None" # Some Result Should be Printed With the 
            return render_template('display.html', status = 200, result = res)

        except: 
            pass 
    return redirect(url_for('home'))  # Redirect to the home page
    # return render_template('index.html', status=500, res = "Internal Server Error ")

if __name__ == "__main__":
    app.run(debug =True)