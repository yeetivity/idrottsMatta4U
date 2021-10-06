"""
Processing of IMU Movesense data from a JSON file to 
usable dictionaries and finally visible graphs
"""

import matplotlib
from matplotlib.pyplot import figure
matplotlib.use("TKAgg")
import numpy as np
import os
import glob
from data_read import DataRead
from data_plot import DataPlot
from data_process import DataProcess
from settings import Settings as s
import time as t

start_time = t.time()

"""
------------------------------LOADING DATA ------------------------------
"""
# Load all the JSON files - Automatic reading from a folder
data = DataRead()                                                     # Initialise the datareading class                                                              # Define path to the folder
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
combAcc = processed_data.combineAccelerations()
simple_kalAcc = processed_data.simpleKalmanFilter()
kalData = processed_data.complexKalmanFilter()

emwaData = processed_data.emwaFilter(combAcc[s.experiment],0.85) #Choose data you want to apply EMWA filter on, and choose alpha value

processed_data.stepRegistration(combAcc[s.experiment])

"""
------------------------------PLOTTING DATA ------------------------------
"""

# Plot combAcc data and raw accx data in 1 plot
data_plot = DataPlot()
accPlot = data_plot.plot1by1(all_csv_data[s.experiment]['time_a'], combAcc[s.experiment], lab='combined acceleration')
accPlot = data_plot.plot1by1(all_csv_data[s.experiment]['time_a'], all_csv_data[s.experiment]['accX'], lab='raw x acceleration', figure=accPlot, linenumber=6)
# Finish the automatic scaling --> automatic determination of ranges in plot1by1 function, also for multiple lines. 
data_plot.show_plot(accPlot, [0,all_csv_data[s.experiment]['time_a'][-1]], [ (combAcc[s.experiment].min() - s.padding),(combAcc[s.experiment].max() + s.padding)],
                     'magnitude', 'timestamp', title='Combined acceleration and raw x acceleration', legend=True)

# Plot combAcc data and raw acc data in subplots
accSubPlot = data_plot.plot2by1(all_csv_data[s.experiment]['time_a'], combAcc[s.experiment], 
                                all_csv_data[s.experiment]['time_a'], all_csv_data[s.experiment]['accX'],
                                lab1 = 'combAcc', lab2= 'accX')
accSubPlot = data_plot.plot2by1(all_csv_data[s.experiment]['time_a'], all_csv_data[s.experiment]['accY'],
                                all_csv_data[s.experiment]['time_a'], all_csv_data[s.experiment]['accZ'],
                                lab1= 'accY', lab2='accZ', figure= accSubPlot, subplotnumber=1,
                                linenumber1 = 1, linenumber2 = 4)
# Add automatic scaling
data_plot.show_plot(accSubPlot, [0,20000], [-10, 30],
                    'magnitude', 'timestamp', title='Combined acceleration and raw accelerations', legend=True)

# Plot all accelerations in seperate subplots
accSubSubPlot = data_plot.plot2by2( all_csv_data[s.experiment]['time_a'], combAcc[s.experiment], 
                                    all_csv_data[s.experiment]['time_a'], all_csv_data[s.experiment]['accX'],
                                    all_csv_data[s.experiment]['time_a'], all_csv_data[s.experiment]['accY'],
                                    all_csv_data[s.experiment]['time_a'], all_csv_data[s.experiment]['accZ'],
                                    lab1= 'combAcc', lab2= 'accX', lab3='accY', lab4='accZ')
data_plot.show_plot(accSubSubPlot, [0,20000], [-10, 30],
                    'magnitude', 'timestamp', title='Combined acceleration and raw accelerations', legend=True)

# Plot kalman filter
KalvsCom = data_plot.plot2by1( all_csv_data[s.experiment]['time_a'], combAcc[s.experiment], 
                    all_csv_data[s.experiment]['time_a'], simple_kalAcc[s.experiment],
                    lab1 = 'combAcc', lab2= 'kalman filtered combAcc')
data_plot.show_plot(KalvsCom, [0,20000], [-10, 30],
                    'magnitude', 'timestamp', title='Combined acceleration and raw accelerations', legend=True)

# Plot complex kalman filter
KalComplex = data_plot.plot3by1(    all_csv_data[s.experiment]['time_a'], kalData[s.experiment][0], 
                                    all_csv_data[s.experiment]['time_a'], kalData[s.experiment][1],
                                    all_csv_data[s.experiment]['time_a'], kalData[s.experiment][2],
                                    lab1= 'position', lab2 ='speed', lab3='acceleration')
data_plot.show_plot(KalComplex, [0,20000], [-10, 30],
                    'magnitude', 'time', title='Complex Kalman Filter results', legend=True)

# Plot EMWA filter
emwaPlot = data_plot.plot1by1(all_csv_data[s.experiment]['time_a'], emwaData, lab='EMWA filtered combined acceleration')
emwaPlot = data_plot.plot1by1(all_csv_data[s.experiment]['time_a'], combAcc[s.experiment], lab='combined acceleration', figure=emwaPlot, linenumber=6)
data_plot.show_plot(emwaPlot, [0,20000], [-10, 30],
                    'magnitude', 'time', title='EMWA filter', legend=True)
"""
------------------------------CODE ENDING ------------------------------
"""
print("--- %s seconds ---" % (t.time()- start_time))