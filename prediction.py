from __future__ import print_function
import matplotlib.pyplot as plt
import Functions as l4f
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import numpy as np
import pickle
from joblib import Parallel, delayed 
import joblib 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob, os
from tensorflow import keras

def getActivityPercentage(filepath):
    # import data
    column_names_df = [
        'x',
        'y',
        'z',
        't',
        'activity'
        ]

    data = pd.read_csv(filepath, names = column_names_df, skiprows= 25, skipfooter = 25, engine = 'python')

    # prepare dataset
    TIME_STEPS = 130
    STEP = 14


    X_test,y_test = l4f.create_dataset(
        data[['x','y','z']],
        data.activity,
        TIME_STEPS,
        STEP
    )


    # encoding of the classes - transform them into binary
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    enc = enc.fit(y_test)

    y_test = enc.transform(y_test)


    fitness_band_model = joblib.load('model.pkl') 

    y_pred = enc.inverse_transform(fitness_band_model.predict(X_test))


    # Calculating classifications
    y_pred = y_pred.flatten()
    total_activities_done = len(y_pred)
    stand_count = np.sum(y_pred == 'Standing')
    lying_count = np.sum(y_pred == 'Lying')
    walking_count = np.sum(y_pred == 'Walking')
    run_count = np.sum(y_pred == 'Running')

    stand_per = (stand_count/total_activities_done) * 100
    lying_per = (lying_count/total_activities_done) * 100
    walking_per = (walking_count/total_activities_done) * 100
    running_per = (run_count/total_activities_done) * 100

    print(stand_per)
    print(lying_per)
    print(walking_per)
    print(running_per)

    return {
        "StandingPercentage":round(stand_per,2),
        "LyingPercentage":round(lying_per,2),
        "WalkingPercentage":round(walking_per,2),
        "RunningPercentage":round(running_per,2)
    }



