from matplotlib import pyplot as plt
import numpy as np

class DataPlot(object):
    """
    Class to plot the data that is imported from DataRead
    """

    def __init__(self):
        """
        =INPUT=
        readdata:
            readdata is an array filled with the dictionaries that define the data
        
        =NOTES=
            In the JSON files the gyrometer data is somehow called magnetometer data, therefore we correct this mistake here by renaming mag into gyr
        """

        self.colors = ( (0, 0, 0.99608),
                        (0, 0.50196, 0.99608),
                        (0, 0.99608, 0.99608),
                        (0.50196, 0.99608, 0.50196),
                        (0.99608, 0.99608, 0),
                        (0.99608, 0.50196, 0),
                        (0.99608, 0, 0),
                        (0.50196, 0, 0))                                # Some non standard colors Todo: reconfigure with more 'space' between colors
        self.marker = ('o' , 'v', 'x', '+', 'd')                        # Some markers

        return
    
    
    def plot1by1(self, xdata, ydata, lab='', figure=None, linenumber=1):
        """
        =INPUT=
        xdata       data that should be plotted on x axis
        ydata       data that should be plotted on y axis
        label       label that should be given to the data
        figure      figure data should be plotted on, initialized to be none
        linenumber  number of the line you are plotting

        =OUTPUT=
        figure      keeps the current figure
        """

        # Initializing the figure to be plotted to
        if figure is None:
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)

        else:
            ax = figure.axes[0]
        
        # Plotting the data
        ax.plot(xdata, ydata, color=self.colors[linenumber-1], linestyle= '-', label = lab)

        return figure


    def kalAccPlot(self, Data, figure=None):
        if (figure == None):
            figure, ax = plt.subplots(1, 3)
            time = np.insert(self.time[0], 0, 0)
            ax[0].plot(time, Data[0][2])
            ax[1].plot(time, Data[0][1])
            ax[2].plot(time, Data[0][0])
        else:
            ax = figure.axes[0]

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

