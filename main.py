"""
Processing IMU data and visualising sprint data
"""

import numpy as np
import time as t
import matplotlib
matplotlib.use("Qt5Agg")

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

combAcc = processed_data.combineAccelerations()                     # Combine accelerations
emwaData = processed_data.emwaFilter(combAcc[s.experiment],0.85)    # Apply EMWA filter to combined accelerations and use alpha=0.85
accKalData = processed_data.complexKalmanFilter(combAcc)            # Apply kalman filter to combined acceleration

peaks, valleys = processed_data.stepRegistration(combAcc[s.experiment]) # Find the peaks and valleys of the combined acceleration



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
# accPlot = data_plot.plot1by1(data[s.experiment]['time_a'], combAcc[s.experiment], lab='combined acceleration')
# accPlot = data_plot.plot1by1(data[s.experiment]['time_a'], data[s.experiment]['accX'], lab='raw x acceleration', figure=accPlot, colornumber=6)
# data_plot.show_plot(accPlot, [0,data[s.experiment]['time_a'][-1]], [ (combAcc[s.experiment].min() - s.padding),(combAcc[s.experiment].max() + s.padding)],
#                      'magnitude', 'timestamp', title='Combined acceleration and raw x acceleration', legend=True)


# Plot all accelerations and combined acceleration
accSubSubPlot = data_plot.plot2by2( data[s.experiment]['time_a'], combAcc[s.experiment], 
                                    data[s.experiment]['time_a'], data[s.experiment]['accX'],
                                    data[s.experiment]['time_a'], data[s.experiment]['accY'],
                                    data[s.experiment]['time_a'], data[s.experiment]['accZ'],
                                    lab1= 'combAcc', lab2= 'accX', lab3='accY', lab4='accZ')
data_plot.show_plot(accSubSubPlot, [0,20000], [-10, 30],
                    'magnitude', 'timestamp', title='Combined acceleration and raw accelerations', legend=True)


# Plot complex kalman filtered data
KalComplex = data_plot.plot3by1(    data[s.experiment]['time_a'], accKalData[s.experiment][0], 
                                    data[s.experiment]['time_a'], accKalData[s.experiment][1],
                                    data[s.experiment]['time_a'], accKalData[s.experiment][2],
                                    lab1= 'position', lab2 ='speed', lab3='acceleration')
data_plot.show_plot(KalComplex, [0,20000], [-10, 30],
                    'magnitude', 'time', title='Complex Kalman Filter results', legend=True)

# Plot EMWA filter
emwaPlot = data_plot.plot1by1(data[s.experiment]['time_a'], emwaData, lab='EMWA filtered combined acceleration')
emwaPlot = data_plot.plot1by1(data[s.experiment]['time_a'], combAcc[s.experiment], lab='combined acceleration', figure=emwaPlot, colornumber=6)
emwaPlot = data_plot.plot1by1(peaks[1], peaks[0], lab="peaks", figure=emwaPlot, colornumber=2, points=True)
emwaPlot = data_plot.plot1by1(valleys[1], valleys[0], lab="valleys", figure=emwaPlot, colornumber=3, points=True)
data_plot.show_plot(emwaPlot, [0,20000], [-10, 30],
                    'magnitude', 'time', title='EMWA filter', legend=True)

# # Plot kalman filtered gyro data
# KalGyrX = data_plot.plot2by1(    data[s.experiment]['time_a'], gyroKalDataX[s.experiment][0], 
#                                 data[s.experiment]['time_a'], gyroKalDataX[s.experiment][1],
#                                 lab1 = 'angle', lab2= 'angular speed')
# data_plot.show_plot(KalGyrX, [0,20000], [-40, 70],
#                     'magnitude', 'timestamp', title='Kalman filtered angular velocity and position around x axis', legend=True)

# KalGyrY = data_plot.plot2by1(    data[s.experiment]['time_a'], gyroKalDataY[s.experiment][0], 
#                                 data[s.experiment]['time_a'], gyroKalDataY[s.experiment][1],
#                                 lab1 = 'angle', lab2= 'angular speed')
# data_plot.show_plot(KalGyrY, [0,20000], [-100, 50],
#                     'magnitude', 'timestamp', title='Kalman filtered angular velocity and position around y axis', legend=True)

# KalGyrZ = data_plot.plot2by1(    data[s.experiment]['time_a'], gyroKalDataZ[s.experiment][0], 
#                                 data[s.experiment]['time_a'], gyroKalDataZ[s.experiment][1],
#                                 lab1 = 'angle', lab2= 'angular speed')
# data_plot.show_plot(KalGyrZ, [0,20000], [-50, 50],
#                     'magnitude', 'timestamp', title='Kalman filtered angular velocity and position around z axis', legend=True)


# Plot horizontal component of acceleration
HorAcc = data_plot.plot1by1(data[s.experiment]['time_a'], horCompo[s.experiment], lab='horizontal component of acceleration')
data_plot.show_plot(HorAcc, [0,20000], [-10, 30],
                    'magnitude', 'timestamp', title='horizontal component of acceleration', legend=True)


"""
------------------------------CODE ENDING ------------------------------
"""
print("--- %s seconds ---" % (t.time()- start_time))