"""
Processing of IMU Movesense data from a JSON file to 
usable dictionaries and finally visible graphs
"""

import matplotlib
# maybe I need to add matplotlib.use("TkAGG") (install TKinter package)
import numpy as np
import os
from json_importer import DataProcesser as dp
import time as t

start_time = t.time()

# Load all the JSON files - Manual pathing way
data = dp()
f5 = data.load_json(path='JSON_DATA/6DOF-footright5.json')
f12_5 = data.load_json(path='JSON_DATA/6DOF-footright12_5.json')
f20 = data.load_json(path='JSON_DATA/6DOF-footright20.json')
fsprint = data.load_json(path='JSON_DATA/6DOF-footrightsprint.json')

l5 = data.load_json(path='JSON_DATA/6DOF-legright5.json')
l12_5 = data.load_json(path='JSON_DATA/6DOF-legright12_5.json')
l20 = data.load_json(path='JSON_DATA/6DOF-legright20.json')
lsprint = data.load_json(path='JSON_DATA/6DOF-legrightsprint.json')

# Load all the JSON files - Automatic reading from a folder NOT FINISHED
path_to_json = 'JSON_DATA/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

# Plot the raw data

# Process the raw data with filters and such

# Plot the processed data

print("--- %s seconds ---" % (t.time()- start_time))