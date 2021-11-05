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
from kalman_filter import KalmanFilter as kf

start_time = t.time()

"""
------------------------------LOADING DATA ------------------------------
"""
# Initialise the datareading class
Data = DataRead()
rawdata = Data.read(s.folderpath, filetype=s.filetype) # array [experimentnr.][acc/gyro/timestamp][timestamp.data]
timestamps = rawdata[s.experiment]['time_a']

"""
------------------------------PROCESSING DATA ------------------------------
"""
# Initialise the data processing class - overwrite data reading class
Data = DataProcess(rawdata[s.experiment])

# Combine the x, y and z accelerations 
comb_acc = Data.combineAccelerations()

# Register the steps in the combined acceleration
peaks, valleys, indices = Data.stepRegistration()

# Filter the combined accelerations with a EMWA filter
emwa_data = Data.emwaFilter(comb_acc, alpha=0.85)

# Create vectors in fixed coordinate frame
vector_data = Data.vectorComponents()

# Initialise the kalmanfilter class
kf_comb_acc = kf(comb_acc, Type='Acc')
kf_anterioposterior = kf(vector_data[0], Type='Acc')
kf_mediolateral = kf(vector_data[1], Type='Acc')
kf_vertical = kf(vector_data[2], Type='Acc')

# Compute the kalmanfiltered vectors
pos, vel, acc = kf_comb_acc.kalmanFilter(reset_times=indices)
pos_ML, vel_ML, acc_ML = kf_mediolateral.kalmanFilter(reset_times=peaks)
pos_AP, vel_AP, acc_AP = kf_anterioposterior.kalmanFilter(reset_times=peaks)
pos_vert, vel_vert, acc_vert = kf_vertical.kalmanFilter(reset_times=peaks)

"""
------------------------------PLOTTING DATA ------------------------------
"""
# Initialise the data plotting class
Data_plot = DataPlot()                          #Todo: different axis lables in subplots

# Create figure with combined accelerations
accPlot = Data_plot.plot1by2(xdata1=timestamps, ydata1=comb_acc, lab1='combined accelerations', linenumber1=0,
                             xdata2=timestamps, ydata2=rawdata[s.experiment]['accX'], lab2='x acceleration', linenumber2=0 )
accPlot = Data_plot.plot1by2(figure=accPlot, xdata1=timestamps, ydata1=emwa_data, lab1='EMWA filtered combined acceleration', linenumber1=1, subplotnumber=0 )
accPlot = Data_plot.plot1by2(figure=accPlot, xdata1=timestamps, ydata1=rawdata[s.experiment]['accY'], lab1='y acceleration', linenumber1=1, subplotnumber=1,
                            xdata2=timestamps, ydata2=rawdata[s.experiment]['accZ'], lab2='z acceleration', linenumber2=2)
accPlot = Data_plot.plot1by2(figure=accPlot, xdata1=peaks[1], ydata1=peaks[0], lab1='peaks and valleys', points=True, linenumber1=1,
                            xdata2=valleys[1], ydata2=valleys[0], linenumber2=1, subplotnumber=0 )

# Plot figure with combined accelerations
Data_plot.show_plot(accPlot, y_label='acceleration [m/s]', x_label=['time [ms]'], title='Accelerations', legend=True)

# Create figure with accelerations, velocities and positions found from combined accelerations
combPlot = Data_plot.plot3by1(  xdata1=timestamps, ydata1=pos, lab1='position',
                                xdata2=timestamps, ydata2=vel, lab2='velocity',
                                xdata3=timestamps, ydata3=acc, lab3='acceleration')

# Plot figure with accelerations, velocities and positions
Data_plot.show_plot(combPlot, y_label='', x_label='time [ms]', title='Processed combined accelerations', legend=True)

# Create figure with accelerations, velocities and positions found from ML vector #! Under Construction
combPlot = Data_plot.plot3by1(  xdata1=timestamps, ydata1=pos_ML, lab1='position',
                                xdata2=timestamps, ydata2=vel_ML, lab2='velocity',
                                xdata3=timestamps, ydata3=acc_ML, lab3='acceleration')

# Plot figure with accelerations, velocities and positions
Data_plot.show_plot(combPlot, y_label='', x_label='time [ms]', title='Processed AP accelerations', legend=True)

"""
------------------------------CODE ENDING ------------------------------
"""
print("--- %s seconds ---" % (t.time()- start_time))