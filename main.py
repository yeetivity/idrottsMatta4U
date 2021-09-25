"""
Processing of IMU Movesense data from a JSON file to 
usable dictionaries and finally visible graphs
"""

import matplotlib
matplotlib.use("TkAGG")
import numpy as np
import os
from data_read import DataRead
from data_plot import DataPlot
import time as t

start_time = t.time()

# Load all the JSON files - Automatic reading from a folder
data = DataRead()                                                                                   # Initialise the datareading class
folderpath = 'JSON_DATA/Experiment1/'                                                             # Define path to the folder
json_files = [pos_json for pos_json in os.listdir(folderpath) if pos_json.endswith('.json')]      # Read all the seperate paths for every .json file
all_data = data.load_all(folderpath ,paths = json_files)                                                        # Use load_all function to create an array with all data

# Plot the raw data
data_plot = DataPlot(all_data)
accXplot = data_plot.AccComparePlot()
positionXplot = data_plot.SensorPositionComparePlot()

data_plot.show_plot(figure=accXplot, x_lim=[0,250], y_lim=[-500,500], y_label= 'magnitude', x_label='sample number',
                    title= 'X accelerations for different speeds', legend=True)
data_plot.show_plot(figure=positionXplot, x_lim=[0,250], y_lim=[-500,500], y_label= 'magnitude', x_label='sample number',
                    title= 'X accelerations for different sensor positions', legend=True)                  

# Process the raw data with filters and such

# Plot the processed data

print("--- %s seconds ---" % (t.time()- start_time))