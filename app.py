# Model Prediction and Activity Percentage

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import stepcount
from joblib import Parallel, delayed 
import joblib 
import matplotlib.pyplot as plt
import Functions as l4f
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob, os
from tensorflow import keras
import prediction
from flask_cors import CORS, cross_origin

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls','csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/analysis', methods=['POST'])
@cross_origin(origin='*')
def calculate():
    if 'activityRawFile' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['activityRawFile']
    
    bodyWeight = request.form.get('weight')
    name = request.form.get('name')
    age = request.form.get('age')
    height = request.form.get('height')

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        

        
        # Preprocessing
        predictedActivityResult = prediction.getActivityPercentage(filepath)
        stepCountResult = stepcount.step_count(filepath , bodyWeight)
        result = {
            "name":name,
            "weight":bodyWeight,
            "height":height,
            "age":age,
            "Duration":stepCountResult["Duration"],
            "CalorieBurnt":stepCountResult["CalorieBurnt"],
            "Distance":stepCountResult["Distance"],
            "Steps":stepCountResult["Steps"],
            "StandingPercentage":predictedActivityResult["StandingPercentage"],
            "LyingPercentage":predictedActivityResult["LyingPercentage"],
            "WalkingPercentage":predictedActivityResult["WalkingPercentage"],
            "RunningPercentage":predictedActivityResult["RunningPercentage"]
        }

        return jsonify(result)
    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)

