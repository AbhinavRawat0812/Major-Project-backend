import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sps

def step_count(filepath , bodyWeight):
    column_names = [
        'x',
        'y',
        'z',
        't',
        'activity'
    ]

    df=pd.read_csv(filepath)
    time = df.t/1000

    fig,ax = plt.subplots()
    plt.title('Labelled Data')
    ax.plot(time, df.x)
    ax.plot(time, df.y)
    ax.plot(time, df.z)
    ax.set_ylabel("Accelerations (G)")
    ax.set_xlabel("Time (s)")
    ax.legend('xyz', loc='upper center')
    ax2 = ax.twinx()
    ax2.plot(time,df.activity,color='black')

    # Calculate sampling frequency
    sfreq = 1000/np.mean(np.diff(df.t))

    # Isolate walking and running data
    stepdata = df.loc[(df.activity == "Running")|(df.activity == "Walking")]
    stepdata = stepdata.reset_index(drop=True)

    stepdata_time = stepdata.t/1000


    # Filter running and walking data
    low_pass =3 #critical Frequency
    low_pass1 = low_pass/(sfreq/2) #sfreq/2 is the nyquist fq
    b2, a2 =sps.butter(4, low_pass1, btype='lowpass')
    xfilt = sps.filtfilt(b2,a2,stepdata.x)

    xfilt_peaks, _ = sps.find_peaks(xfilt) #peaks of filtered running/walking
    STEPS = len(xfilt_peaks)


    # plt.figure()
    # plt.plot(stepdata_time,xfilt)
    # plt.plot(stepdata_time[xfilt_peaks], xfilt[xfilt_peaks], "x", color='g')
    # plt.title('Peaks of Filtered Data x-axis')
    # plt.ylabel("Acceleration [G]")
    # plt.xlabel('Time [s]')
    # plt.legend()
    # plt.show()

    MET = 3.0      #Metabolic equivalent of task
    DURATION = 1179  #In seconds
    print(type(float(bodyWeight)))
    BODY_WEIGHT = float(bodyWeight)
    calorie_burnt = (MET * 3.5 * ((BODY_WEIGHT*2.2)/200)) * DURATION/60

    DISTANCE = (0.78 * STEPS)/1000


    # print("TIME = ",DURATION, "Min")
    # print("BODY WEIGHT = ", BODY_WEIGHT, "Kg")
    # print("Steps = ", STEPS)
    # print("CALORIE BURNT = ",calorie_burnt,"Kcal")
    # print("Distance Covered = ", DISTANCE,"Km")

    return {
            "Duration":DURATION,
            "CalorieBurnt":round(calorie_burnt,2),
            "Distance":round(DISTANCE,2),
            "Steps":STEPS
            }

