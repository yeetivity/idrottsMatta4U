"""
Processing of IMU Movesense data from a JSON file to 
usable dictionaries and finally visible graphs
"""

import matplotlib
from matplotlib.pyplot import figure
matplotlib.use("TkAGG")
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




"""
------------------------------PLOTTING DATA ------------------------------
"""

# Plot the raw data
data_plot = DataPlot()
accPlot = data_plot.plot1by1(all_csv_data[experiment]['time_a'], combAcc[experiment], lab='combined acceleration')
accPlot = data_plot.plot1by1(all_csv_data[experiment]['time_a'], all_csv_data[experiment]['accX'], lab='raw x acceleration', figure=accPlot, linenumber=6)

# Finish the automatic scaling --> automatic determination of ranges in plot1by1 function, also for multiple lines. 
data_plot.show_plot(accPlot, [0,all_csv_data[experiment]['time_a'][-1]], [ (combAcc[experiment].min() - padding),(combAcc[experiment].max() + padding)],
                     'magnitude', 'timestamp', title='Combined acceleration and raw x acceleration', legend=True)


# accXplot = data_plot.AccComparePlot()
# positionXplot = data_plot.SensorPositionComparePlot()
# # combPlot = data_plot.kalAccPlot(combAcc)
# simplekalAccplot = data_plot.simple_kalAccPlot(simple_kalAcc)
# simplekalAccPlot = data_plot.simple_kalAccPlot(combAcc, figure= simplekalAccplot)

# kalAccPlot = data_plot.kalAccPlot(kalData)



# data_plot.show_plot(figure=accXplot, x_lim=[0,250], y_lim=[-500,500], y_label= 'magnitude', x_label='sample number',
#                     title= 'X accelerations for different speeds', legend=True)
# data_plot.show_plot(figure=positionXplot, x_lim=[0,250], y_lim=[-500,500], y_label= 'magnitude', x_label='sample number',
#                     title= 'X accelerations for different sensor positions', legend=True) 
# data_plot.show_plot(figure=simplekalAccplot, x_lim=[0, 20000], y_lim=[-50,50], y_label= 'magnitude', x_label='timestamp',
#                     title= 'kalman Filter', legend=True)     
# data_plot.show_plot(figure=kalAccPlot, x_lim=[0, 20000], y_lim=[-50,50], y_label= 'magnitude', x_label='timestamp',
#                     title= 'complex kalman Filter', legend=True)            

# Process the raw data with filters and such

# Plot the processed data

print("--- %s seconds ---" % (t.time()- start_time))