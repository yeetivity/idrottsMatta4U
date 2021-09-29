from matplotlib import pyplot as plt
import numpy as np

class DataPlot(object):
    """
    Class to plot the data that is imported from DataRead
    """

    def __init__(self,readdata):
        """
        =INPUT=
        readdata:
            readdata is an array filled with the dictionaries that define the data
        
        =NOTES=
            In the JSON files the gyrometer data is somehow called magnetometer data, therefore we correct this mistake here by renaming mag into gyr
        """
        self.storage = readdata

        # Initialise empty arrays
        self.accX = []
        self.accY = []
        self.accZ = []

        self.gyrX = []
        self.gyrY = []
        self.gyrZ = []

        self.time = []

        self.samples = []

        # Fill the arrays
        for i in range(len(readdata)):
            self.accX.append(readdata[i]['accX'])
            self.accY.append(readdata[i]['accY'])
            self.accZ.append(readdata[i]['accZ'])

            self.gyrX.append(readdata[i]['gyrX'])
            self.gyrY.append(readdata[i]['gyrY'])
            self.gyrZ.append(readdata[i]['gyrZ'])

            self.time.append(readdata[i]['time_a'])

        for ii in range(len(readdata[0]['accX'])):
            self.samples.append(ii + 1) # append the sample number to the list, first sample is 1

        for j in range (len(self.time)):
            self.time[j] = np.array(self.time[j])
            self.time[j] = self.time[j] - self.time[j][0]


        return

    def AccComparePlot(self, figure=None):

        # Initialise the figure
        figure, ax = plt.subplots(2, 2)

        #Plot data on the figure
        ax[0, 0].plot(self.time[0], self.accX[0], color='blue', label='5km/h')
        ax[0, 1].plot(self.time[1], self.accX[1], color='red', label='12.5km/h')
        ax[1, 0].plot(self.time[2], self.accX[2], color='green', label='20km/h')
        ax[1, 1].plot(self.time[3], self.accX[3], color='purple', label='sprint')

        return figure

    def SensorPositionComparePlot(self, figure=None):

        # Initialise the figure
        figure, ax = plt.subplots(2, 2)

        #Plot data on the figure
        ax[0, 0].plot(self.time[0], self.accX[0], color='blue', label='5km/h, sensor on foot')
        ax[0, 1].plot(self.time[4], self.accX[4], color='blue', linestyle= '--', label='5km/h, sensor on leg')
        ax[1, 0].plot(self.time[1], self.accX[1], color='green', label='sprint, sensor on foot')
        ax[1, 1].plot(self.time[5], self.accX[5], color='green', linestyle= '--', label='sprint, sensor on leg')

        return figure

    def kalAccPlot(self, kalAcc_Data ,figure=None):
        figure, ax = plt.subplots(1, 1)

        ax.plot(self.time[0], kalAcc_Data[0], color='red', label='Kalman Filtered data, 1D')

        return figure

    def show_plot(self, figure, x_lim, y_lim, y_label, x_label, title, backgroundcolor=(0.827, 0.827, 0.827), legend=False):
        """
        =INPUT=
        figure:
            The figure created in plot
        _lim:
            array with max and min axis value
        _label:
            string that will become axis label
        title:
            string that will become figure title
        backgroundcolor:
            array of RGB 0-1 values
            initialised to be white (1,1,1)

        =NOTES=
        Legend is now overly full, however does show what marker/color is what
        ToDo: find way to create more generalized legend
        """
        for i in range(len(figure.axes)):
            ax = figure.axes[i]
            subplottitles = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

            if len(figure.axes) != 1:
                figure.suptitle(title)
                ax.set_title(subplottitles[i])

            else:
                ax.set_title(title)

            #Add figure labels
            ax.set_ylabel(y_label)
            ax.set_xlabel(x_label)

            #Add axis definitions
            ax.set_ylim(y_lim)
            ax.set_xlim(x_lim)

            #Change background color
            ax.set_facecolor(backgroundcolor)

            #Add gridlines
            ax.grid()

            #Add legend
            if (legend is True):
                figure.legend(loc=8, ncol=4)

            #Show figure
            figure.show()
            
        return

