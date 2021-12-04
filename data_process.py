from matplotlib.pyplot import step
import numpy as np
from settings import Settings as s
from kalman_filter import KalmanFilter as kf
import copy

def find_nearest(array,value):
    idx = (np.abs(array - value)).argmin()
    return idx

def resetTimeList(time_list):
    first_time = time_list[0]

    for i in range(len(time_list)):
        time_list[i]=time_list[i]-first_time

    return time_list

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
        K0 = 300        # Initial time interval threshold of Ki
        alpha = 0.7     # Scale factor used to determine the time interval threshold
        W2 = 3          # Number of consecutive valleys
        TH_pk = 25       # Peak detection threshold to exclude false detection
        TH_s = 200      # Fixed value to detect static states and determine whether to stop the update of K_i
        
        TH = 6          # Statistical value that used to distinguish the state of motion is intense or gentle  
        W1 = 3          # The window size of the acceleration-magnitude detector
        TH_vy = 3     # Valley detection threshold that utilized to detect the valleys 
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
        
        #* Valid valley detection
        # 1. Minima detection
        for i in range(1,len(self.combAcc)-1):
            if ((self.combAcc[i] < self.combAcc[i+1]) and (self.combAcc[i] < self.combAcc[i-1]) and (self.combAcc[i] < TH_vy)):
                minima[0].append(self.combAcc[i])
                minima[1].append(self.time[i])

            # 1. Maxima Detection
            if ((self.combAcc[i] > self.combAcc[i+1]) and (self.combAcc[i] > self.combAcc[i-1]) and (self.combAcc[i] > TH_pk)):
                maxima[0].append(self.combAcc[i])
                maxima[1].append(self.time[i])
                indices.append(i)

        #. Adapt TH_PK threshold according to the maximum value and remove all the wrong maxima
        TH_pk = max(maxima[0])*0.4 #TODO Choose the right value
        ind = 0
        while ind < len(maxima[0]):
            if maxima[0][ind] <= TH_pk:
                maxima[0].pop(ind)
                maxima[1].pop(ind)
                indices.pop(ind)
                
                ind = ind
            else :
                ind+=1

        #. Remove all the valleys before first peak
        ind = 0
        while (ind<len(minima[0]) and minima[1][ind]<maxima[1][0]) :
            if minima[1][ind]<maxima[1][0]:
                minima[1].pop(ind)
                minima[0].pop(ind)
                
                ind = ind
            else:
                ind+=1
            
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

            #* Valid valley detection
            if ((minima[1][max(n,1)]-minima[1][max(n-1,0)]) < Ki):
                index = minima[0].index(max([minima[0][max(n,1)],minima[0][max(n-1,0)]]))     
                minima[0].pop(index)                                                          
                minima[1].pop(index)   

        j = 1
        while j < len(maxima[0]):
            if ((maxima[1][j]-maxima[1][j-1]) < K0):
                if maxima[0][j]>maxima[0][j-1]:
                    index = j-1     # Determine the index of the smallest peak
                else :
                    index = j
                maxima[0].pop(index)                                            # Delete smallest peak
                maxima[1].pop(index)
                j = j
            else:
                j+= 1

        # Valid peak detection + valid valley detection
        ind=1
        while ind<len(maxima[0]):
            mini = minima[1][ind-1]
            maxi = maxima[1][ind]
            maxi_1 = maxima[1][ind-1]
            
            #remove if multiple valleys in one cycle
            ind_extra_valley = ind
            while ind_extra_valley<len(minima[1]) and minima[1][ind_extra_valley]<=maxima[1][ind] :
                minima[1].pop(ind)
                minima[0].pop(ind)

                ind_extra_valley = ind_extra_valley

            if ind == 1:
                if mini<=maxi and mini>=maxi_1:
                    ind+=1
                else:
                    maxima[0].pop(ind)
                    maxima[1].pop(ind)
                    indices.pop(ind)

                    ind=ind
            else :
                if mini<=maxi and mini>=maxi_1:
                    ind+=1
                else:
                    maxima[0].pop(ind-1)
                    maxima[1].pop(ind-1)
                    indices.pop(ind-1)

                    ind=ind

        if len(maxima[0])<len(minima[0]):
            difference = len(minima[0])-len(maxima[0])
            print(difference)
            for i in range(difference):
                minima[0].pop(len(maxima[0]))
                minima[1].pop(len(maxima[0]))

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
        copyTimeList = self.time.copy()

        oneStepTimeList = copyTimeList[indices[step_number]:indices[step_number+1]]
        return oneStepData, oneStepTimeList


    def stepFrequency(self, peaks):
        """
        """
        # Compute step frequency for individual steps
        stepFreq = []
        for i in range(len(peaks[0])-1):
            timeOneStep = (peaks[1][i+1] - peaks[1][i]) / 1000
            stepFreq.append(1 / timeOneStep)

        # Compute Average Step Frequency
        avgStepFreq = sum(stepFreq) / len(stepFreq)            

        return avgStepFreq, stepFreq 
        


