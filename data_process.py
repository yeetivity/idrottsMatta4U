from sys import intern
import numpy as np
from math import factorial
from settings import Settings as s
from scipy.signal import find_peaks
from kalman_filter import KalmanFilter as kf

def find_nearest(array,value):
    idx = (np.abs(array - value)).argmin()
    return idx

class DataProcess(object):
    """
    
    """
    def __init__(self, storeddata): #TODO make everything that can be array an array
        self.gyro = [storeddata['gyrX'],storeddata['gyrY'],storeddata['gyrZ']]
        self.acc = [storeddata['accX'],storeddata['accY'],storeddata['accZ']]
        self.time = storeddata['time_a'] 

        self.combAcc = []
        return

    
    def combineAccelerations(self):
        """
    Function that combines the accelerations
        """
        self.combAcc =  (np.sqrt(np.square(self.acc[0]) + 
                                 np.square(self.acc[1]) +
                                 np.square(self.acc[2]))
                                 - s.gravity )
        return self.combAcc


    def stepRegister_Init(self):
        #Todo See if we can move this to the settings somehow
        K0 = 200        # Initial time interval threshold of Ki
        alpha = 0.7     # Scale factor used to determine the time interval threshold
        W2 = 5          # Number of consecutive valleys
        TH_pk = 5       # Peak detection threshold to exclude false detection
        TH_s = 190      # Fixed value to detect static states and determine whether to stop the update of K_i
        
        TH = 6          # Statistical value that used to distinguish the state of motion is intense or gentle  
        W1 = 3          # The window size of the acceleration-magnitude detector
        TH_vy = 1.9     # Valley detection threshold that utilized to detect the valleys 
        return K0, alpha, W2, TH_pk, TH_s, TH, W1, TH_vy


    def stepRegistration(self):
        """
        #Todo See if we can move this to kalmanfilter class --> Answer == no
        """
        #* INITIALIZATION
        K0, alpha, W2, TH_pk, TH_s, TH, W1, TH_vy = self.stepRegister_Init()

        maxima = [[],[]]        # Array with maxima
        minima = [[],[]]        # Array with minima
        indices = []

        T=[]

        for i in range(len(self.combAcc)-W1-1):
            T.append(0)
            variance = np.var(self.combAcc[i+1:i+1+W1])
            for k in range(i+1,i+1+W1):
                T[i]=T[i]+np.square(self.combAcc[k]-s.gravity)*(1/(np.square(variance)*W1))
        
        #* Valid valley detection
        # 1. Minima detection
        for i in range(1,len(self.combAcc)-1-W1):
            if ((self.combAcc[i] < self.combAcc[i+1]) and (self.combAcc[i] < self.combAcc[i-1]) and (T[i] < TH_vy)):
                minima[0].append(self.combAcc[i])
                minima[1].append(self.time[i])

            # 1. Maxima Detection
            if ((self.combAcc[i] > self.combAcc[i+1]) and (self.combAcc[i] > self.combAcc[i-1]) and (self.combAcc[i] > TH_pk)):
                maxima[0].append(self.combAcc[i])
                maxima[1].append(self.time[i])
                indices.append(i)
        
        # 2. Single valley detection with temporal threshold constraint
        for i in range(1,len(self.combAcc)-1):

            t_i = self.time[i]
            n = find_nearest(np.asarray(minima[1]),t_i)
            t_n = minima[1][n]

            if ((np.abs(t_i-t_n)) < TH_s):
                if (minima[1][n]-minima[1][max(0,n-W2)])==0:
                    Ki = K0
                else :
                    Ki = alpha*(minima[1][n]-minima[1][max(0,n-W2)])/W2
            elif (np.abs(t_i - t_n) >= TH_s):
                Ki = K0 

            #* Valid peak detection
            if ((minima[1][max(n,1)]-minima[1][max(n-1,0)]) < Ki):
                index = minima[0].index(max([minima[0][max(n,1)],minima[0][max(n-1,0)]]))     # Determine the index of the smallest peak
                minima[0].pop(index)                                                          # Delete smallest peak
                minima[1].pop(index)   

            n_max = find_nearest(np.asarray(maxima[1]),t_i)

            # 2. Single Peak Detection with temporal threshold constraint
            if ((maxima[1][max(n_max,1)]-maxima[1][max(n_max-1,0)]) < Ki):
                index = maxima[0].index(min([maxima[0][max(n_max,1)],maxima[0][max(n_max-1,0)]]))     # Determine the index of the smallest peak
                maxima[0].pop(index)                                                                  # Delete smallest peak
                maxima[1].pop(index)
                indices.pop(index)   

        #Results
        print('Amount of peaks:', len(maxima[0]))
        print('Amount of valleys:', len(minima[0]))
        return maxima, minima, indices


    def emwaFilter(self,data,alpha):
        """
        EMWA filter
        """
        #Initialization
        emwaData = [data[0]]

        #Filtering
        for k in range(1, len(data)):
            emwaData.append(alpha*emwaData[k-1]+(1-alpha)*data[k])
            
        return emwaData


    def vectorComponents(self):
        """
        Compute the horizontal component of acceleration (or velocity) #Todo: check if right axes are used.
        =INPUT=
        =OUTPUT=
        self.horComp    array with: [experiment][timestamp]
        """
        # Initialise a coor_fxd_acc
        coor_fxd_acc = [[],[],[]]
        # We need angles, we compute these by kalmanfiltering the gyro data
        AnglesX = kf(self.gyro[0], self.acc, Type='Gyro')
        angle = AnglesX.kalmanFilter(direction='y')

        #! Since we already have an accX, accY, accZ as output, do we even need this?
        AnglesY = kf(self.gyro[1], self.acc, Type='Gyro')
        AnglesZ = kf(self.gyro[2], self.acc, Type='Gyro')

        #takes one column of self.acc for every experiment
        coor_lcl_acc = np.array(    [self.acc[0],
                                    self.acc[1],
                                    self.acc[2]])

        # all the values in one experiment
        for j in range(len(coor_lcl_acc[0])):
            # Todo: check if kalmangyro outputs radians
            rotation_matrix = np.array([    [np.cos(angle[0][j]),    0,  -np.sin(angle[0][j])],
                                            [0,                         1,     0],
                                            [np.sin(angle[0][j]),    0,  np.cos(angle[0][j])]])
            array = (np.dot(rotation_matrix, coor_lcl_acc[:,[j]]))
            for i in range(3):
                coor_fxd_acc[i].append(array[i][0])

            # Todo substract gravity in z direction
            
        return coor_fxd_acc

    def dataOneStep(self, full_data, indices, step_number):
        """
        Cuts full data into a list corresponding to one step
        """

        oneStepData = full_data[indices[step_number]:indices[step_number+1]]
        oneStepTimeList = self.time[indices[step_number]:indices[step_number+1]]
        return oneStepData, oneStepTimeList



    




    def GCT1(self,maxima,minima):
        #init
        found = False
        i=0
        gct = []
        #make a list of maxima and minima pairs (only their timestamps)
        #first maximum is just reference so it starts at index 1
        # sync
        minStart = 0
        maxStart = 1
        totTime = len(maxima[0]) - 1

        while not found:
            if (abs(maxima[1][maxStart]-minima[1][minStart])<250):
                found = True
            else :
                minStart += 1

        # select synced arrays
        timepairs = [maxima[1][1:], minima[1][minStart:minStart + totTime]]
        #calculate GCT
        for a,b in zip(*timepairs):
            gct.append(abs(a - b))
        return gct


    """
    Sliding window signal on selected data
    

    """
    def SW(self, width: int, signal_type: str):
        if signal_type == 'x':
            signal = self.acc[0]
        elif signal_type == 'y':
            signal = self.acc[1]
        elif signal_type == 'z':
            signal = self.acc[2]
        elif signal_type == 'comb':
            signal = self.combAcc
        else:
            print('only x, y, z or comb is allowed')
        #signal = signal[experiment_n]

        noise_signal = np.zeros(len(signal))

        signal_arr = np.array(signal + [0]*width)
        print(f'size: {signal_arr.size}')
        for i in range(len(signal)):# len - width -1 + width -1
            
            noise_signal[i] = np.sum(np.abs(signal_arr[i] - signal_arr[np.arange(i+1, i+width)]))
            #noise_signal[i] = np.abs(signal_arr[i] - signal_arr[i + width - 1])
        

        filtered_signal = savitzky_golay(noise_signal, 81, 2)

        ordered_signal = np.sort(filtered_signal)

        scndmaximum = ordered_signal[-1]
        return filtered_signal/scndmaximum



def gct_peaks(filtered_signal,threshold=0.8):

    peaks_idx, _ = find_peaks(filtered_signal, distance=50, height=0.4)
    final_peaks_idx = []
    for i in range(peaks_idx.size-1):
        if filtered_signal[peaks_idx[i]]>threshold:
            final_peaks_idx.append(peaks_idx[i])
            final_peaks_idx.append(peaks_idx[i+1])

    return final_peaks_idx

def gct_from_peaks(peaks_idx: list, signal, time):
    # if peaks_idx.size % 2 != 0:
    #     peaks_idx = peaks_idx[:-1]
    # if not np.all(signal[peaks_idx[::2]] > signal[peaks_idx[1::2]]):
    #     peaks_idx = peaks_idx[1:]
    #     peaks_idx = peaks_idx[:-1]
    # assert(np.all(signal[peaks_idx[::2]] > signal[peaks_idx[1::2]]))
    return time[peaks_idx[1::2]] - time[peaks_idx[::2]]
            


def savitzky_golay(y, window_size, order, deriv=0, rate=1) -> np.ndarray:
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


    def stepFrequency(self, peaks):
        """
        """
        # Compute Average Step Frequency
        nb_peaks = len(peaks[0])                        # number of steps
        total_time = (peaks[1][-1] - peaks[1][0]) / 1000    # total time between steps
        avgStepFreq = nb_peaks / total_time             # average frequency

        # Compute step frequency for individual steps
        stepFreq = []
        for i in range(len(peaks[0])-1):
            timeOneStep = (peaks[1][i+1] - peaks[1][i]) / 1000
            stepFreq.append(1 / timeOneStep)

        return avgStepFreq, stepFreq 
