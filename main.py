"""
Processing of IMU Movesense data from a JSON file to 
usable dictionaries and finally visible graphs
"""

import matplotlib
matplotlib.use("TkAGG")
import numpy as np
import os
import glob
from data_read import DataRead
from data_plot import DataPlot
from data_process import DataProcess
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
# ERROR, THIS DOES NOT READ IN ORDER --> USE GLOB?
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
processed_data.combineAccelerations()
processed_data.simpelKalmanFilter()



"""
------------------------------PLOTTING DATA ------------------------------
"""

# Plot the raw data
data_plot = DataPlot(all_json_data)
accXplot = data_plot.AccComparePlot()
positionXplot = data_plot.SensorPositionComparePlot()

data_plot.show_plot(figure=accXplot, x_lim=[0,250], y_lim=[-500,500], y_label= 'magnitude', x_label='sample number',
                    title= 'X accelerations for different speeds', legend=True)
data_plot.show_plot(figure=positionXplot, x_lim=[0,250], y_lim=[-500,500], y_label= 'magnitude', x_label='sample number',
                    title= 'X accelerations for different sensor positions', legend=True)                  

# Process the raw data with filters and such

# Plot the processed data

print("--- %s seconds ---" % (t.time()- start_time))