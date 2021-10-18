"""
Processing of IMU Movesense data from a JSON file to 
usable dictionaries and finally visible graphs
"""

import matplotlib
from matplotlib.pyplot import figure
matplotlib.use("TkAgg")
import numpy as np
import os
import glob
from data_read import DataRead
from data_plot import DataPlot
from data_process import DataProcess
import time as t

start_time = t.time()

# A few settings
experiment = 0
padding = 5             # Determines how much space automatic scaling plots have

"""
------------------------------LOADING DATA ------------------------------
"""
# Load all the JSON files - Automatic reading from a folder
data = DataRead()                                                     # Initialise the datareading class
paths = sorted(glob.glob('JSON_DATA/Experiment1/'+ "/*.json"))        # Read all the seperate paths for every .json file
all_json_data = data.load_jsons(paths)                                # Use load_all function to create an array with all data

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
combAcc = processed_data.combineAccelerations()             # Combine accelerations
accKalData = processed_data.complexKalmanFilter(combAcc)    # Kalman filter combined acceleration

# Kalman filtered gyro data    
gyroKalDataX = processed_data.complexKalmanFilterGyro(processed_data.gyroX, processed_data.kalGyroX, processed_data.pitch)
gyroKalDataY = processed_data.complexKalmanFilterGyro(processed_data.gyroY, processed_data.kalGyroY, processed_data.roll)
gyroKalDataZ = processed_data.complexKalmanFilterGyro(processed_data.gyroZ, processed_data.kalGyroZ, processed_data.yaw)

horCompo = processed_data.horizontalComponent(gyroKalDataX) # Horizontal component of acceleration

"""
------------------------------PLOTTING DATA ------------------------------
"""

# Plot combined acceleration and raw x acceleration #Todo finish automatic scaling
data_plot = DataPlot()
accPlot = data_plot.plot1by1(all_csv_data[experiment]['time_a'], combAcc[experiment], lab='combined acceleration')
accPlot = data_plot.plot1by1(all_csv_data[experiment]['time_a'], all_csv_data[experiment]['accX'], lab='raw x acceleration', figure=accPlot, linenumber=6)
data_plot.show_plot(accPlot, [0,all_csv_data[experiment]['time_a'][-1]], [ (combAcc[experiment].min() - padding),(combAcc[experiment].max() + padding)],
                     'magnitude', 'timestamp', title='Combined acceleration and raw x acceleration', legend=True)


# Plot all accelerations and combined acceleration
accSubSubPlot = data_plot.plot2by2( all_csv_data[experiment]['time_a'], combAcc[experiment], 
                                    all_csv_data[experiment]['time_a'], all_csv_data[experiment]['accX'],
                                    all_csv_data[experiment]['time_a'], all_csv_data[experiment]['accY'],
                                    all_csv_data[experiment]['time_a'], all_csv_data[experiment]['accZ'],
                                    lab1= 'combAcc', lab2= 'accX', lab3='accY', lab4='accZ')
data_plot.show_plot(accSubSubPlot, [0,20000], [-10, 30],
                    'magnitude', 'timestamp', title='Combined acceleration and raw accelerations', legend=True)


# Plot complex kalman filtered data
KalComplex = data_plot.plot3by1(    all_csv_data[experiment]['time_a'], accKalData[experiment][0], 
                                    all_csv_data[experiment]['time_a'], accKalData[experiment][1],
                                    all_csv_data[experiment]['time_a'], accKalData[experiment][2],
                                    lab1= 'position', lab2 ='speed', lab3='acceleration')
data_plot.show_plot(KalComplex, [0,20000], [-10, 30],
                    'magnitude', 'time', title='Complex Kalman Filter results', legend=True)


# Plot kalman filtered gyro data
KalGyrX = data_plot.plot2by1(    all_csv_data[experiment]['time_a'], gyroKalDataX[experiment][0], 
                                all_csv_data[experiment]['time_a'], gyroKalDataX[experiment][1],
                                lab1 = 'angle', lab2= 'angular speed')
data_plot.show_plot(KalGyrX, [0,20000], [-40, 70],
                    'magnitude', 'timestamp', title='Kalman filtered angular velocity and position around x axis', legend=True)

KalGyrY = data_plot.plot2by1(    all_csv_data[experiment]['time_a'], gyroKalDataY[experiment][0], 
                                all_csv_data[experiment]['time_a'], gyroKalDataY[experiment][1],
                                lab1 = 'angle', lab2= 'angular speed')
data_plot.show_plot(KalGyrY, [0,20000], [-100, 50],
                    'magnitude', 'timestamp', title='Kalman filtered angular velocity and position around y axis', legend=True)

KalGyrZ = data_plot.plot2by1(    all_csv_data[experiment]['time_a'], gyroKalDataZ[experiment][0], 
                                all_csv_data[experiment]['time_a'], gyroKalDataZ[experiment][1],
                                lab1 = 'angle', lab2= 'angular speed')
data_plot.show_plot(KalGyrZ, [0,20000], [-50, 50],
                    'magnitude', 'timestamp', title='Kalman filtered angular velocity and position around z axis', legend=True)


# Plot horizontal component of acceleration
HorAcc = data_plot.plot1by1(all_csv_data[experiment]['time_a'], horCompo[experiment], lab='horizontal component of acceleration')
data_plot.show_plot(HorAcc, [0,20000], [-10, 30],
                    'magnitude', 'timestamp', title='horizontal component of acceleration', legend=True)


"""
------------------------------CODE ENDING ------------------------------
"""
print("--- %s seconds ---" % (t.time()- start_time))