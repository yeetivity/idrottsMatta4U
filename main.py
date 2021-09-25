"""
Processing of IMU Movesense data from a JSON file to 
usable dictionaries and finally visible graphs
"""

import matplotlib
# maybe I need to add matplotlib.use("TkAGG") (install TKinter package)
import numpy as np
import os
from data_read import DataRead
from data_plot import DataPlot
import time as t

start_time = t.time()

# Load all the JSON files - Automatic reading from a folder
data = DataRead()                                                                                   # Initialise the datareading class
path_to_json = 'JSON_DATA/'                                                                         # Define path to the folder
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]      # Read all the seperate paths for every .json file
all_data = data.load_all(paths = json_files)                                                        # Use load_all function to create an array with all data

# Plot the raw data
data_plot = DataPlot(all_data)

# Process the raw data with filters and such

# Plot the processed data

print("--- %s seconds ---" % (t.time()- start_time))