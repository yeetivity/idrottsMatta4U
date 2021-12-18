from math import exp
from matplotlib import pyplot as plt
import numpy as np

from data_process import DataProcess, gct_from_peaks, gct_peaks

class DataPlot(object):
    """
    Class to plot the data
    """

    def __init__(self):
        """
        """

        self.colors = ( (0.61176, 0.04314, 0.95686),                    # Purple
                        (0.00000, 0.00000, 0.00000),                    # Black
                        (0.48235, 0.82745, 0.53725),                    # Emerald
                        (0.37255, 0.65882, 0.82745),                    # Carolina Blue
                        (0.05098, 0.12941, 0.63137),                    # Blue Pantone
                        (0.24706, 0.05098, 0.07059))                    # Dark Sienna

        self.marker = ('o' , 'v', 'x', '+', 'd')                        # Some markers

        return
    
    
    def plot1by1(self, xdata, ydata, lab='', figure=None, cnr=1, points=False, mnr=1):
        """
        =INPUT=
        xdata       data that should be plotted on x axis
        ydata       data that should be plotted on y axis
        label       label that should be given to the data
        figure      figure data should be plotted on, initialized to be none
        cnr         number of the color that you want to use
        points      if True, plot points instead of line
        mnr         number of the marker you want to use


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
            ax.plot(xdata, ydata, color=self.colors[cnr%6], linestyle= '-', label = lab)

        if points is True:
            ax.plot(xdata, ydata, marker=self.marker[mnr%5], linestyle= 'None',
                markersize=8, markerfacecolor=(1, 1, 1, 0), markeredgecolor=self.colors[cnr%6])

        return figure


    def plot1by2(   self, xdata1, ydata1, xdata2=None, ydata2=None, lab1 = '', lab2 = '',
                    figure=None, subplotnumber=None, linenumber1=0, linenumber2=1, points=False):
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
            figure.axes[0].plot(xdata1, ydata1, color=self.colors[linenumber1], linestyle= '-', label = lab1)
            figure.axes[1].plot(xdata2, ydata2, color=self.colors[linenumber2], linestyle= '-', label = lab2)                                # 2 Data in 2 plots
        
        elif (subplotnumber != None):
            if points is False:
                figure.axes[subplotnumber].plot(xdata1, ydata1, color=self.colors[linenumber1], linestyle= '--', label = lab1)
                figure.axes[subplotnumber].plot(xdata2, ydata2, color=self.colors[linenumber2], linestyle= '--', label = lab2)          # 2 Data in 1 plot
            if points is True:
                figure.axes[subplotnumber].plot(xdata1, ydata1, marker=self.marker[linenumber1%5], linestyle='None',
                markersize=8, markerfacecolor=(1,1,1,0), markeredgecolor=self.colors[linenumber1%6], label = lab1)
                figure.axes[subplotnumber].plot(xdata2, ydata2, marker=self.marker[linenumber2], linestyle='None',
                markersize=8, markerfacecolor=(1,1,1,0), markeredgecolor=self.colors[linenumber2%6], label = lab2)
        
        else:
            pass
        
        return figure
        

    def plot2by1(   self, xdata1, ydata1, xdata2=None, ydata2=None, lab1 = '', lab2 = '',
                    figure=None, subplotnumber=None, linenumber1=1, linenumber2=2):
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
                    linenumber1=1, linenumber2=2, linenumber3=3):
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


    def show_plot(self, figure, y_label, x_label, title, backgroundcolor=(0.827, 0.827, 0.827), legend=False,  x_lim=None, y_lim=None):
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
            initialised to be grey (1,1,1)

        =NOTES=
        """
        for i in range(len(figure.axes)):
            ax = figure.axes[i]
            subplottitles = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

            if len(figure.axes) != 1:
                figure.suptitle(title)
                #ax.set_title(subplottitles[i])

            else:
                ax.set_title(title)

            #Add figure labels
            ax.set_ylabel(y_label)
            ax.set_xlabel(x_label)

            #Add axis definitions
            if y_lim is not None:
                ax.set_ylim(y_lim)
            if x_lim is not None:
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

def plot_gct_total(rawdata):
    experiment_lut = [
        ['long_jump', 1],
        ['long_jump', 2],
        ['long_jump', 3],
        ['long_jump', 4],
        ['sprint', 1],
        ['sprint', 2],
        ['sprint', 3],
        ['high_jump', 1],
        ['high_jump', 2],
        ['high_jump', 3],
        ['high_jump', 4],
        ['80m', 1],
        ['80m', 2],
        ['80m', 3],
    ]

    speeds = {
        1: 0,
        2: 1,
        3: 2,
        4: 3
    }

    name_lut = {
        'long_jump': 0,
        'sprint': 1,
        'high_jump': 2,
        '80m': 3
    }

    colors = {
        'long_jump': 'red',
        'sprint': 'blue',
        'high_jump': 'green',
        '80m': 'black'
    }
    
    gct_total = []

    fig, ax = plt.subplots(dpi=100)
    for e in range(14):
       exp_n = e
       data = DataProcess(rawdata[exp_n])
       noise_signal = data.SW(50, 'x')
       peaks_idx = gct_peaks(noise_signal)
       gct_total.append(gct_from_peaks(peaks_idx, noise_signal, data.time))
       x = np.array([experiment_lut[e][1]] * len(gct_total[-1]))
       ax = plt.scatter(x=x + np.random.normal(0, 0.3, x.size),
        y=gct_total[-1],
        c=colors[experiment_lut[e][0]], alpha=0.3)

    # ax.set_ylabel('GCT')
    # ax.set_xlabel('Speed')
    # ax.set_title('Ground contact time scatter plot')
    # ax.legend()
    # fig.show()

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    axes = [ax1, ax2, ax3, ax4]

    for e in range(14):
       axes[speeds[experiment_lut[e][1]]].plot(
           gct_total[e],
           c=colors[experiment_lut[e][0]]
        )
    
    axes[2].set_ylim(0, 300)
    axes[0].legend(['first'] + ['second'] + ['third'] + ['fourth'])
    axes[1].legend(['first'] + ['second'] + ['third'])
    axes[2].legend(['first'] + ['second'] + ['third'] + ['fourth'])
    axes[3].legend(['first'] + ['second'] + ['third'])
    
    axes[0].set_title('long jump')
    axes[1].set_title('sprint')
    axes[2].set_title('high jump')
    axes[3].set_title('80m')

    for ax in axes:
        ax.set_xlabel('steps') 
        ax.set_ylabel('GCT [ms]') 



    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
    return fig2