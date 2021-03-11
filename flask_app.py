import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model
from flask import Flask, render_template, request


with open("model_pickle.pkl",'rb') as m:
    model = pickle.load(m)


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    varience = float(int_features[0])
    skewness = float(int_features[1])
    curtosis = float(int_features[2])
    entropy = float(int_features[3])
    
    prediction = model.predict([[varience,skewness,curtosis,entropy]])

    final_result = []
    if prediction[0]==1:
        final_result.append("NOTE IS AUTHENTIC")
    else:
        final_result.append("NOTE IS NOT AUTHENTIC")
    return render_template("index.html",prediction = final_result[0])


if __name__ == "__main__":

    app.run()
