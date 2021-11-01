"""
Processing IMU data and visualising sprint data
"""

import numpy as np
import time as t
import matplotlib
matplotlib.use("TkAgg")

from settings import Settings as s
from data_read import DataRead
from data_plot import DataPlot
from data_process import DataProcess

start_time = t.time()

"""
------------------------------LOADING DATA ------------------------------
"""
folderpath = 'CSV_DATA2'
Data = DataRead() # Initialise the datareading class
data = Data.read(folderpath, filetype='csv')

"""
------------------------------PROCESSING DATA ------------------------------
"""
processed_data = DataProcess(data)

# Raw data processing
combAcc = processed_data.combineAccelerations()                                  # Combine accelerations
peaks, valleys, indices = processed_data.stepRegistration(combAcc[s.experiment]) # Find the peaks and valleys of the combined acceleration

# Filtering
emwaData = processed_data.emwaFilter(combAcc[s.experiment],0.85)    # Apply EMWA filter to combined accelerations and use alpha=0.85
accKalData = processed_data.complexKalmanFilter(combAcc[s.experiment], indices)            # Apply kalman filter to combined acceleration

# Kalman filtered gyro data    
gyroKalDataX = processed_data.complexKalmanFilterGyro(processed_data.gyroX, processed_data.kalGyroX, processed_data.pitch)
gyroKalDataY = processed_data.complexKalmanFilterGyro(processed_data.gyroY, processed_data.kalGyroY, processed_data.roll)
gyroKalDataZ = processed_data.complexKalmanFilterGyro(processed_data.gyroZ, processed_data.kalGyroZ, processed_data.yaw)

horCompo = processed_data.horizontalComponent(gyroKalDataX) # Horizontal component of acceleration

"""
------------------------------PLOTTING DATA ------------------------------
"""
data_plot = DataPlot()  # Initialise class
#Todo: different axis lables in subplots

# Plot all accelerations and combined acceleration
accSubSubPlot = data_plot.plot2by2( data[s.experiment]['time_a'], combAcc[s.experiment], 
                                    data[s.experiment]['time_a'], data[s.experiment]['accX'],
                                    data[s.experiment]['time_a'], data[s.experiment]['accY'],
                                    data[s.experiment]['time_a'], data[s.experiment]['accZ'],
                                    lab1= 'combAcc', lab2= 'accX', lab3='accY', lab4='accZ')
data_plot.show_plot(accSubSubPlot, x_lim=[0,20000], y_lim=[-10, 30],
                    y_label='magnitude', x_label='time', title='Combined acceleration and raw accelerations', legend=True)


# Plot complex kalman filtered data
KalComplex = data_plot.plot3by1(    data[s.experiment]['time_a'], accKalData[0], 
                                    data[s.experiment]['time_a'], accKalData[1],
                                    data[s.experiment]['time_a'], accKalData[2],
                                    lab1= 'position', lab2 ='speed', lab3='acceleration')
data_plot.show_plot(KalComplex, x_lim=[0,20000], y_lim=[-10, 30],
                    y_label='magnitude', x_label='time', title='Position, speed and acceleration', legend=True)

# Plot EMWA filter
emwaPlot = data_plot.plot1by1(data[s.experiment]['time_a'], emwaData, lab='EMWA filtered combined acceleration')
emwaPlot = data_plot.plot1by1(data[s.experiment]['time_a'], combAcc[s.experiment], lab='combined acceleration', figure=emwaPlot, cnr=6)
emwaPlot = data_plot.plot1by1(peaks[1], peaks[0], lab="peaks", figure=emwaPlot, cnr=2, mnr=1, points=True)
emwaPlot = data_plot.plot1by1(valleys[1], valleys[0], lab="valleys", figure=emwaPlot, cnr=3, mnr=1, points=True)
data_plot.show_plot(emwaPlot, x_lim=[0,20000], y_lim=[-10, 30],
                    y_label='magnitude', x_label='time', title='EMWA filtered accelerations', legend=True)


# Plot horizontal component of acceleration
HorAcc = data_plot.plot1by1(data[s.experiment]['time_a'], horCompo[s.experiment], lab='horizontal component of acceleration')
data_plot.show_plot(HorAcc, x_lim=[0,20000], y_lim=[-10, 30],
                    y_label='magnitude', x_label='time', title='Horizontal acceleration', legend=True)


"""
------------------------------CODE ENDING ------------------------------
"""
print("--- %s seconds ---" % (t.time()- start_time))