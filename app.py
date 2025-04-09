from flask import Flask, render_template, url_for, request
import pandas as pd

import pickle
# import numpy as np


#Model load
loaded_model = pickle.load(open('/home/hariom/Project/AQI-Predictor/Deployement/random_forest_regression_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')
 # Optional HTML form

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('real_2018.csv') # data as df
    my_prediction = loaded_model.predict(df.iloc[:,:-1].values)  # Ensure same order
    my_prediction = my_prediction.tolist()

    print("Prediction:", my_prediction)

    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
