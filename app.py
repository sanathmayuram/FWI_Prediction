from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

ridge_model = pickle.load(open('Ridge.pkl', 'rb'))
standed_scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/',methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        Temparature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled =  standed_scaler.transform([[Temparature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        result = ridge_model.predict(new_data_scaled)

        return render_template("home.html",results=result[0])

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")