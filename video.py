import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from data_read import DataRead
from data_plot import DataPlot
from data_process import DataProcess
from settings import Settings as s
import glob
import cv2

"""
------------------------------LOADING DATA ------------------------------
"""
data = DataRead() 
# Load all CSV acc files - Automatic reading from a folder
paths = sorted(glob.glob('CSV_DATA/ACC/'+ "/*.csv"))
all_csv_accdata = data.load_csvs(paths)
data.transform_csvformat(all_csv_accdata, acc=True)

# Load all CSV gyr files - Automatic reading from a folder
paths = sorted(glob.glob('CSV_DATA/GYRO/'+ "/*.csv"))
all_csv_gyrdata = data.load_csvs(paths)
all_csv_data = data.transform_csvformat(all_csv_gyrdata, acc=False)

"""
------------------------------PROCESSING DATA ------------------------------
"""
processed_data = DataProcess(all_csv_data)
combAcc = processed_data.combineAccelerations()
simple_kalAcc = processed_data.simpleKalmanFilter()
kalData = processed_data.complexKalmanFilter()

emwaData = processed_data.emwaFilter(combAcc[s.experiment],0.85) #Choose data you want to apply EMWA filter on, and choose alpha value

peaks, valleys = processed_data.stepRegistration(combAcc[s.experiment])

"""
------------------------------CREATING FRAMES ------------------------------
"""

#Getting timestamps for each frame of the video

cap = cv2.VideoCapture('video_10_foot.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]

while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        
    else:
        break

cap.release()
print(len(timestamps))

#Init
first_peak = peaks[0][0]
to = peaks[1][0]

data_plot = DataPlot()
plt.plot(all_csv_data[s.experiment]['time_a'], combAcc[s.experiment], label='combined acceleration')
plt.plot(to, first_peak, 'ro')
plt.xlim([to-2500,to+2500])
plt.legend()
plt.savefig('Plt_img/img_0.png')
plt.close()

#Plot a red dot on the combined acceleration graph corresponding to the frame in the video

def find_nearest(array,value):
    idx = (np.abs(array - value)).argmin()
    return idx

for i in range(1,len(timestamps)):
    tg = to + timestamps[i]
    ind_t = find_nearest(all_csv_data[s.experiment]['time_a'],tg)
    time_data = all_csv_data[s.experiment]['time_a'][ind_t]
    plt.plot(all_csv_data[s.experiment]['time_a'], combAcc[s.experiment], label='combined acceleration')
    plt.plot(time_data, combAcc[s.experiment][ind_t], 'ro')
    plt.xlim([time_data-2500,time_data+2500])
    plt.legend()
    plt.savefig('Plt_img/img_'+str(i)+'.png')
    plt.close()