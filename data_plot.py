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
    
    
    def plot1by1(self, xdata, ydata, lab='', figure=None, colornumber=1, points=False):
        """
        =INPUT=
        xdata       data that should be plotted on x axis
        ydata       data that should be plotted on y axis
        label       label that should be given to the data
        figure      figure data should be plotted on, initialized to be none
        colornumber number of the line you are plotting

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
        if points is False:
            ax.plot(xdata, ydata, color=self.colors[colornumber-1], linestyle= '-', label = lab)

        if points is True:
            ax.plot(xdata, ydata, marker=self.marker[colornumber], linestyle= 'None',
                markersize=8, markerfacecolor=(1, 1, 1, 0), markeredgecolor=self.colors[colornumber])

        return figure


    def plot1by2(   self, xdata1, ydata1, xdata2=None, ydata2=None, lab1 = '', lab2 = '',
                    figure=None, subplotnumber=None, linenumber1=0, linenumber2=1):
        """
        =INPUT=
        xdata       data that should be plotted on x axis
        ydata       data that should be plotted on y axis
        lab         label that should be given to the data
        figure      figure data should be plotted on, initialized to be none

        =OUTPUT=
        figure      keeps the current figure
        """

        # Initializing the figure to be plotted to
        if figure is None:
            figure = plt.figure()
            ax1 = figure.add_subplot(1, 2, 1)
            ax2 = figure.add_subplot(1, 2, 2)

        else:
            ax1 = figure.axes[0]
            ax2 = figure.axes[1]
        
        # Plotting the data
        if (((xdata2 is None) and (ydata2 is None)) and (subplotnumber == None)):
            ax1.plot(xdata1, ydata1, color=self.colors[linenumber1], linestyle='-', label = lab1)                                  # 1 Data in first plot
        
        elif (((xdata2 is None) and (ydata2 is None)) and (subplotnumber != None)):
            figure.axes[subplotnumber].plot(xdata1, ydata1, color=self.colors[linenumber1], linestyle='-', label = lab1)           # 1 Data in chosen plot
        
        elif (subplotnumber == None):
            figure.axes[0].plot(xdata1, ydata1, color=self.colors[0], linestyle= '-', label = lab1)
            figure.axes[1].plot(xdata2, ydata2, color=self.colors[7], linestyle= '-', label = lab2)                                # 2 Data in 2 plots
        
        elif (subplotnumber != None):
            figure.axes[subplotnumber].plot(xdata1, ydata1, color=self.colors[linenumber1], linestyle= '-', label = lab1)
            figure.axes[subplotnumber].plot(xdata2, ydata2, color=self.colors[linenumber2], linestyle= '-', label = lab2)          # 2 Data in 1 plot
        
        else:
            pass
        
        return figure
        

    def plot2by1(   self, xdata1, ydata1, xdata2=None, ydata2=None, lab1 = '', lab2 = '',
                    figure=None, subplotnumber=None, linenumber1=0, linenumber2=1):
        """
        =INPUT=
        xdata       data that should be plotted on x axis
        ydata       data that should be plotted on y axis
        lab         label that should be given to the data
        figure      figure data should be plotted on, initialized to be none

        =OUTPUT=
        figure      keeps the current figure
        """

        # Initializing the figure to be plotted to
        if figure is None:
            figure = plt.figure()
            ax1 = figure.add_subplot(2, 1, 1)
            ax2 = figure.add_subplot(2, 1, 2)

        else:
            ax1 = figure.axes[0]
            ax2 = figure.axes[1]
        
        # Plotting the data
        if (((xdata2 is None) and (ydata2 is None)) and (subplotnumber == None)):
            ax1.plot(xdata1, ydata1, color=self.colors[linenumber1], linestyle='-', label = lab1)                                  # 1 Data in first plot
        
        elif (((xdata2 is None) and (ydata2 is None)) and (subplotnumber != None)):
            figure.axes[subplotnumber].plot(xdata1, ydata1, color=self.colors[linenumber1], linestyle='-', label = lab1)           # 1 Data in chosen plot
        
        elif (subplotnumber == None):
            figure.axes[0].plot(xdata1, ydata1, color=self.colors[0], linestyle= '-', label = lab1)
            figure.axes[1].plot(xdata2, ydata2, color=self.colors[7], linestyle= '-', label = lab2)                                # 2 Data in 2 plots
        
        elif (subplotnumber != None):
            figure.axes[subplotnumber].plot(xdata1, ydata1, color=self.colors[linenumber1], linestyle= '-', label = lab1)
            figure.axes[subplotnumber].plot(xdata2, ydata2, color=self.colors[linenumber2], linestyle= '-', label = lab2)          # 2 Data in 1 plot
        
        else:
            pass
        
        return figure


    def plot2by2(   self, xdata1, ydata1, xdata2=None, ydata2=None, xdata3=None, ydata3=None,
                    xdata4=None, ydata4=None, lab1 = '', lab2 = '', lab3= '', lab4= '', figure=None, subplotnumber=None,
                    linenumber1=0, linenumber2=1, linenumber3=2, linenumber4=3):
        """
        =INPUT=
        xdata       data that should be plotted on x axis
        ydata       data that should be plotted on y axis
        lab         label that should be given to the data
        figure      figure data should be plotted on, initialized to be none

        =OUTPUT=
        figure      keeps the current figure
        """

        # Initializing the figure to be plotted to
        if figure is None:
            figure = plt.figure()
            ax1 = figure.add_subplot(2, 2, 1)
            ax2 = figure.add_subplot(2, 2, 2)
            ax3 = figure.add_subplot(2, 2, 3)
            ax4 = figure.add_subplot(2, 2, 4)

        else:
            pass
        
        # Plotting the data
        # 1 Data per 1 subplot
        ax1.plot(xdata1, ydata1, color=self.colors[linenumber1], linestyle='-', label = lab1)
        ax2.plot(xdata2, ydata2, color=self.colors[linenumber2], linestyle='-', label = lab2)
        ax3.plot(xdata3, ydata3, color=self.colors[linenumber3], linestyle='-', label = lab3)
        ax4.plot(xdata4, ydata4, color=self.colors[linenumber4], linestyle='-', label = lab4)

        
        return figure


    def plot3by1(   self, xdata1, ydata1, xdata2=None, ydata2=None, xdata3=None, ydata3=None,
                    lab1 = '', lab2 = '', lab3= '', figure=None, subplotnumber=None,
                    linenumber1=0, linenumber2=1, linenumber3=2):
        """
        =INPUT=
        xdata       data that should be plotted on x axis
        ydata       data that should be plotted on y axis
        lab         label that should be given to the data
        figure      figure data should be plotted on, initialized to be none

        =OUTPUT=
        figure      keeps the current figure
        """

        # Initializing the figure to be plotted to
        if figure is None:
            figure = plt.figure()
            ax1 = figure.add_subplot(3, 1, 1)
            ax2 = figure.add_subplot(3, 1, 2)
            ax3 = figure.add_subplot(3, 1, 3)

        else:
            pass
        
        # Plotting the data
        # 1 Data per 1 subplot
        ax1.plot(xdata1, ydata1, color=self.colors[linenumber1], linestyle='-', label = lab1)
        ax2.plot(xdata2, ydata2, color=self.colors[linenumber2], linestyle='-', label = lab2)
        ax3.plot(xdata3, ydata3, color=self.colors[linenumber3], linestyle='-', label = lab3)

        
        return figure


    def plot1by3(   self, xdata1, ydata1, xdata2=None, ydata2=None, xdata3=None, ydata3=None,
                    lab1 = '', lab2 = '', lab3= '', figure=None, subplotnumber=None,
                    linenumber1=0, linenumber2=1, linenumber3=2):
        """
        =INPUT=
        xdata       data that should be plotted on x axis
        ydata       data that should be plotted on y axis
        lab         label that should be given to the data
        figure      figure data should be plotted on, initialized to be none

        =OUTPUT=
        figure      keeps the current figure
        """

        # Initializing the figure to be plotted to
        if figure is None:
            figure = plt.figure()
            ax1 = figure.add_subplot(1, 3, 1)
            ax2 = figure.add_subplot(1, 3, 2)
            ax3 = figure.add_subplot(1, 3, 3)

        else:
            pass
        
        # Plotting the data
        # 1 Data per 1 subplot
        ax1.plot(xdata1, ydata1, color=self.colors[linenumber1], linestyle='-', label = lab1)
        ax2.plot(xdata2, ydata2, color=self.colors[linenumber2], linestyle='-', label = lab2)
        ax3.plot(xdata3, ydata3, color=self.colors[linenumber3], linestyle='-', label = lab3)

        
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

