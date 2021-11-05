import numpy as np
from settings import Settings as s
from kalman_filter import KalmanFilter as kf

class DataProcess(object):
    """
    
    """
    def __init__(self, storeddata): #TODO make everything that can be array an array
        self.gyro = [storeddata['gyrX'],storeddata['gyrY'],storeddata['gyrZ']]
        self.acc = [storeddata['accX'],storeddata['accY'],storeddata['accZ']]
        self.time = storeddata['time_a'] 

        self.combAcc = []
        
        # self.pitch = np.zeros((len(self.storeddata), len(self.time)))
        # self.roll = np.zeros((len(self.storeddata), len(self.time)))
        # self.yaw = np.zeros((len(self.storeddata), len(self.time)))
        # self.accX_arr = np.zeros((len(self.storeddata), len(self.time)))       #! Can't we make everything an array?
        # self.accY_arr = np.zeros((len(self.storeddata), len(self.time)))
        # self.accZ_arr = np.zeros((len(self.storeddata), len(self.time)))

        # # Check if we want this in initialisation. 
        # for i in range(len(self.storeddata)):
        #     for j in range(len(self.acc[0][i])):
        #         self.accX_arr[i][j] = self.acc[0][i][j]
        #         self.accY_arr[i][j] = self.acc[1][i][j]
        #         self.accZ_arr[i][j] = self.acc[2][i][j]

        #         # Todo put in kalmanfilter class
        #         self.pitch[i][j] = np.tan(self.accX_arr[i][j]/ (np.sqrt(self.accY_arr[i][j]**2 + self.accZ_arr[i][j]**2)))
        #         self.roll[i][j] = np.tan(self.accY_arr[i][j]/ (np.sqrt(self.accX_arr[i][j]**2 + self.accZ_arr[i][j]**2)))
        #         self.yaw[i][j] = np.tan((np.sqrt(self.accX_arr[i][j]**2 + self.accY_arr[i][j]**2))/self.accZ_arr[i][j])
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


    def stepRegistration(self):
        """
        #Todo See if we can move this to kalmanfilter class --> Answer == no
        """
        #* INITIALIZATION
        K0, Ki, alpha, W2, TH_pk, TH_s, TH, W1, TH_vy = self.stepRegister_Init()

        maxima = [[],[]]        # Array with maxima
        minima = [[],[]]        # Array with minima
        indices = []

        #* Valid valley detection
        # 1. Minima detection
        for i in range(1,len(self.combAcc)-1):
            if ((self.combAcc[i] < self.combAcc[i+1]) and (self.combAcc[i] < self.combAcc[i-1]) and (self.combAcc[i] < TH_vy)):
                minima[0].append(self.combAcc[i])
                minima[1].append(self.time[i])
        
        # 2. Single valley detection with temporal threshold constraint
        j = 1
        while j < len(minima[0]):
            if ((minima[1][j]-minima[1][j-1]) < Ki):
                index = minima[0].index(max([minima[0][j],minima[0][j-1]]))     # Determine the index of the smallest peak
                minima[0].pop(index)                                            # Delete smallest peak
                minima[1].pop(index)
                j = j
            else:
                j+= 1

        #* Valid peak detection
        # 1. Maxima Detection
        for i in range(1,len(self.combAcc)-1):
            if ((self.combAcc[i] > self.combAcc[i+1]) and (self.combAcc[i] > self.combAcc[i-1]) and (self.combAcc[i] > TH_pk)):
                maxima[0].append(self.combAcc[i])
                maxima[1].append(self.time[i])
                indices.append(i)

        # 2. Single Peak Detection with temporal threshold constraint
        j = 1
        while j < len(maxima[0]):
            if ((maxima[1][j]-maxima[1][j-1]) < Ki):
                index = maxima[0].index(min([maxima[0][j],maxima[0][j-1]]))     # Determine the index of the smallest peak
                maxima[0].pop(index)                                            # Delete smallest peak
                maxima[1].pop(index)
                indices.pop(index)
                j = j
            else:
                j+= 1

        # Adaptive thresholds determination

        # Adaptive zero-velocity detection

        # Results
        print('Amount of peaks:', len(maxima[0]))
        print('Amount of valleys:', len(minima[0]))
        return maxima, minima, indices

    def stepRegister_Init(self):
        #Todo See if we can move this to the settings somehow
        K0 = 350        # Initial time interval threshold of Ki
        Ki = K0
        alpha = 0.7     # Scale factor used to determine the time interval threshold
        W2 = 5          # Number of consecutive valleys
        TH_pk = 40      # Peak detection threshold to exclude false detection
        TH_s = 190      # Fixed value to detect static states and determine whether to stop the update of K_i
        
        TH = 6          # Statistical value that used to distinguish the state of motion is intense or gentle  
        W1 = 3          # The window size of the acceleration-magnitude detector
        TH_vy = 1.9     # Valley detection threshold that utilized to detect the valleys 
        return K0, Ki, alpha, W2, TH_pk, TH_s, TH, W1, TH_vy


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
        angle = AnglesX.kalmanFilter(direction='x')

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