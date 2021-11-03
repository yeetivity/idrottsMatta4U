import numpy as np
from settings import Settings as s

def find_nearest(array,value):
    idx = (np.abs(array - value)).argmin()
    return idx

class DataProcess(object):
    """
    
    """
    def __init__(self, storeddata): #TODO clean up the init
        self.storeddata = storeddata

        self.gyroX = [] 
        self.gyroY = []
        self.gyroZ = []

        self.accX = []
        self.accY = []
        self.accZ = []
        self.combAcc = []

        self.gravity = s.gravity
        self.dT = s.samplingfreq

        self.simple_kalAcc = [] 
        self.kalData = []

        self.kalAcc = []
        self.kalVel = []
        self.kalPos = []
        self.kalGyroX = []
        self.kalGyroY = []
        self.kalGyroZ = []

        self.time = storeddata[s.experiment]['time_a']
        self.emwaData = []    
        self.horCompo = []
        maxima = []

        for i in range(len(self.storeddata)):
            self.accX.append(self.storeddata[i]['accX'])
            self.accY.append(self.storeddata[i]['accY'])
            self.accZ.append(self.storeddata[i]['accZ'])
            self.gyroX.append(self.storeddata[i]['gyrX'])
            self.gyroY.append(self.storeddata[i]['gyrY'])
            self.gyroZ.append(self.storeddata[i]['gyrZ'])
            maxima.append(len(self.storeddata[i]['time_a']))
        
        maximum = max(maxima)
        self.pitch = np.zeros((len(self.storeddata), maximum))
        self.roll = np.zeros((len(self.storeddata), maximum))
        self.yaw = np.zeros((len(self.storeddata), maximum))
        self.accX_arr = np.zeros((len(self.storeddata), maximum))
        self.accY_arr = np.zeros((len(self.storeddata), maximum))
        self.accZ_arr = np.zeros((len(self.storeddata), maximum))

        for i in range(len(self.storeddata)):
            for j in range(len(self.accX[i])):
                self.accX_arr[i][j] = self.accX[i][j]
                self.accY_arr[i][j] = self.accY[i][j]
                self.accZ_arr[i][j] = self.accZ[i][j]
                self.pitch[i][j] = np.tan(self.accX_arr[i][j]/ (np.sqrt(self.accY_arr[i][j]**2 + self.accZ_arr[i][j]**2)))
                self.roll[i][j] = np.tan(self.accY_arr[i][j]/ (np.sqrt(self.accX_arr[i][j]**2 + self.accZ_arr[i][j]**2)))
                self.yaw[i][j] = np.tan((np.sqrt(self.accX_arr[i][j]**2 + self.accY_arr[i][j]**2))/self.accZ_arr[i][j])

        return

    
    def combineAccelerations(self):
        """
        Function that processes the accelerations
        """
        for i in range(len(self.storeddata)):
            self.combAcc.append(np.sqrt(np.square(self.storeddata[i]['accX']) + 
                                        np.square(self.storeddata[i]['accY']) +
                                        np.square(self.storeddata[i]['accZ']))
                                        - s.gravity )

        return self.combAcc


    def simpleKalmanFilter(self):
        """
        Kalman with one dimension
        """
        for i in range(len(self.combAcc)):
            z = self.combAcc[i]
            P = np.zeros(len(self.combAcc[i]))
            P[0] = 1

            K = np.zeros(len(self.combAcc[i]))

            R = 10
            Q = 0.5

            x = np.zeros(len(z))

            for j in range(len(self.combAcc[i])):
                K[j] = (P[j-1] + Q) / ((P[j-1] + Q) + R)
                x[j] = x[j-1] + K[j] * (z[j] - x[j-1])
                P[j] = (1 - K[j]) * (P[j-1] + Q)

            self.simple_kalAcc.append(x)
        return self.simple_kalAcc


    def complexKalmanFilter(self, data_to_filter, reset_times, B=None, u=None):
        """
        Kalman with multiple dimensions

        =INPUT=
            data_to_filter  takes combined acceleration or anyy other raw acceleration data
            B               init to be none
            u               initialised to be none

        =OUTPUT=
            self.kalData    6x3x1024 array with:
                            [experiment][pos=0, velocity = 1, acceleration = 2][timestamp]
        """
        if (B is None) and (u is None):
            B = np.zeros((3,3))
            u = np.zeros((3,1))
        else:
            pass
        
        if (len(data_to_filter) <= 100):
            for i in range(len(data_to_filter)):
                # Initialisation of filter
                A, P, Q, H, R, x, X = self.resetKalman()
                z = data_to_filter[i]
                y = np.subtract( z[0], np.dot(H,x))         # Comparing predicted value with measurement

                # Filtering
                for j in range (len(z)):
                    
                    # PREDICTION VALUES
                    x = np.dot(A,x) #+ np.dot(B, u)
                    P = np.add( (np.dot(np.dot(A,P), A.transpose())), Q)
                    
                    # MEASUREMENT VALUES
                    y = np.subtract(z[j], np.dot(H, x))
                    K = np.dot( P, H.transpose()) / (np.dot(np.dot(H, P), H.transpose()) + R)

                    # UPDATE X AND P
                    for ii in range(0,3):
                        x[ii] = x[ii] + y*K[ii]

                    P = np.dot(np.subtract(1, np.dot(K,H)), P)
                    X = np.hstack((X, x))

                X = X[:,1:]
                self.kalData.append(X)
        
        elif (len(data_to_filter) >= 100):
            i=0
            A, P, Q, H, R, x, X, z, y = self.resetKalman(data_to_filter, i, X=None)

            for j in range (len(z)):
                if (j == reset_times[i]):
                    A, P, Q, H, R, x, X, z, y = self.resetKalman(data_to_filter, j, X=X)
                    if (i<(len(reset_times)-1)):
                        i +=1

                # PREDICTION VALUES
                x = np.dot(A,x) #+ np.dot(B, u)
                P = np.add( (np.dot(np.dot(A,P), A.transpose())), Q)
                
                # MEASUREMENT VALUES
                y = np.subtract(z[j], np.dot(H, x))
                K = np.dot( P, H.transpose()) / (np.dot(np.dot(H, P), H.transpose()) + R)

                # UPDATE X AND P
                for ii in range(0,3):
                    x[ii] = x[ii] + y*K[ii]

                P = np.dot(np.subtract(1, np.dot(K,H)), P)
                X = np.hstack((X, x))
            
            X = X[:,1:]

            # i=0
            # for jj in range(len(reset_times)):
            #     X = np.delete(X, reset_times[i],1)
            #     if (i<(len(reset_times)-1)):
            #             i +=1

            self.kalData = X

        return self.kalData

    def resetKalman(self, z, index, X=None):
        A = np.array([  [1, self.dT, 0.5 * self.dT**2],
                        [0, 1, self.dT],
                        [0, 0, 1]])                 # State transition matrix
        P = np.array([  [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 10]])                # State covariance matrix
        Q = np.array([  [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0.5]])               # Process noise covariance matrix
        H = np.array([0, 0, 1])                     # Measurement matrix

        # Initiliaze some filter values
        R = 10                                      # Some scalar
                              
        if (index == 0):
            x = np.array([  [0],                    # Position, velocity, acceleration
                            [0],
                            [0]]) 
            X = x
        else:
            x = np.array([  [X[0][index]],
                            [X[1][index]],
                            [X[2][index]]])
            # X = np.hstack((X,x))

        z = z
        y = np.subtract( z[index], np.dot(H, x))         # Comparing predicted value with measurement
        return A, P, Q, H, R, x, X, z, y
        

    def complexKalmanFilterGyro(self, gyro_data, filtered_gyro, control_data, B=None, u=None):      #Todo: Add correction for w acc
        """
        Kalman with multiple dimensions, for gyro

        =INPUT=
            B               init to be none
            u               initialised to be none

        =OUTPUT=
            self.kalGyro    6x2x1 array with:
                            [experiment][angle=0 (degrees), angular velocity = 1][timestamp]
        """
        filtered_gyro = []
        if (B is None) and (u is None):
            u = np.zeros((2,1))
        else:
            pass


        for i in range(len(gyro_data)):
            # setting for gyro Kalman
            A = np.array([  [1, self.dT],     # angle
                            [0, 1]])    # angular v     # State transition matrix
            P = np.array([  [0, 0],
                            [0, 10]])                # State covariance matrix
            Q = np.array([  [0, 0],
                            [0, 0.5]])               # Process noise covariance matrix
            H = np.array([0, 1])                     # Measurement matrix

            B = np.array([ [0, 0],
                           [0, 1]]) #Todo: something needs to be added here according to Elisa

            #u = np.array([[0],
                        #[self.pitch[0][0]]])

            # Initiliaze some filter values
            R = 10                                      # Some scalar
            z = gyro_data[i]
            control = control_data[i]
            x = np.array([  [0],
                            [0]])                       # angle, angular v
            y = np.subtract( z[0], np.dot(H,x))         # Comparing predicted value with measurement
            X = x

            # filtering
            for j in range (len(z)):
                
                # PREDICTION VALUES
                    # x = Ax + Bu
                u = np.array(   [[0],
                                [control[j]]])
                Bu = np.dot(B, u)
                Ax = np.dot(A,x)
                #x = np.dot(A,x) #+ np.dot(B, u)
                for ii in range(1):
                    x[ii] = Ax[ii] + Bu[ii]
                    # P = A P A^T + Q
                P = np.add( (np.dot(np.dot(A,P), A.transpose())), Q)
                
                # MEASUREMENT VALUES
                    # Y = Z - H X
                y = np.subtract(z[j], np.dot(H, x))
                    # K = (P H^T) / ( ( HPH^T) + R)
                K = np.dot( P, H.transpose()) / (np.dot(np.dot(H, P), H.transpose()) + R)

                # UPDATE X AND P
                    # X = X + KY
                for ii in range(0,2):
                    x[ii] = x[ii] + y*K[ii]
                    # P = (1 - KH) P
                P = np.dot(np.subtract(1, np.dot(K,H)), P)
                
                X = np.hstack((X, x))
            X = X[:,1:]
            filtered_gyro.append(X)


        return filtered_gyro

    # function to get the horizontal component #Todo: check if right axes are used. 
    def horizontalComponent(self, angle):
        """
        Horizontal component of acc (or vel)

        =INPUT=
            angle: angles given out by kalman filter on gyro (6x2x1 array:
                        [experiment][angle=0 (degrees), angular velocity = 1][timestamp])

        =OUTPUT=
            self.horCompo    6x1024 array with:
                            [experiment][timestamp]
        """

        # init list with horizontal components
        self.horCompo = []
        
        for i in range(len(angle)):

            #takes one row of self.accZ (one experiment)
            Z = self.combAcc[i]
            self.horCompoNew = []

            # all the values in one experiment
            for j in range(len(Z)):
                #! np.cos takes radians 
                # Todo: check if kalmangyro outputs radians
                self.horCompoNew.append(Z[j] * np.cos(angle[i][0][j])) #[0]=angle, [1]=angular velocity
            
            self.horCompo.append(self.horCompoNew)
            
        return self.horCompo
    
    
    def emwaFilter(self,data,alpha):
        """
        EMWA filter
        """
        #Initialization
        self.emwaData.append(data[0])

        #Filtering
        for k in range(1, len(data)):
            self.emwaData.append(alpha*self.emwaData[k-1]+(1-alpha)*data[k])
            
        return self.emwaData

    def stepRegistration(self, combAcc):
        """
        """
        #* INITIALIZATION
        K0 = 200       # Initial time interval threshold of Ki
        alpha = 0.7     # Scale factor used to determine the time interval threshold
        W2 = 5          # Number of consecutive valleys
        TH_pk = 5      # Peak detection threshold to exclude false detection
        TH_s = 190      # Fixed value to detect static states and determine whether to stop the update of K_i
        
        
        TH = 6          # Statistical value that used to distinguish the state of motion is intense or gentle  
        W1 = 3          # The window size of the acceleration-magnitude detector
        TH_vy = 1.9     # Valley detection threshold that utilized to detect the valleys 

        maxima = [[],[]]        # Array with maxima
        minima = [[],[]]        # Array with minima

        T=[]

        for i in range(len(combAcc)-W1-1):
            T.append(0)
            variance = np.var(combAcc[i+1:i+1+W1])
            for k in range(i+1,i+1+W1):
                T[i]=T[i]+np.square(combAcc[k]-s.gravity)*(1/(np.square(variance)*W1))
        
        #* Valid valley detection
        # 1. Minima detection
        for i in range(1,len(combAcc)-1-W1):
            if ((combAcc[i] < combAcc[i+1]) and (combAcc[i] < combAcc[i-1]) and (T[i] < TH_vy)):
                minima[0].append(combAcc[i])
                minima[1].append(self.time[i])

            # 1. Maxima Detection
            if ((combAcc[i] > combAcc[i+1]) and (combAcc[i] > combAcc[i-1]) and (combAcc[i] > TH_pk)):
                maxima[0].append(combAcc[i])
                maxima[1].append(self.time[i])
        
        # 2. Single valley detection with temporal threshold constraint
        for i in range(1,len(combAcc)-1):

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

            if ((minima[1][max(n,1)]-minima[1][max(n-1,0)]) < Ki):
                index = minima[0].index(max([minima[0][max(n,1)],minima[0][max(n-1,0)]]))     # Determine the index of the smallest peak
                minima[0].pop(index)                                         # Delete smallest peak
                minima[1].pop(index)   

            #* Valid peak detection

            n_max = find_nearest(np.asarray(maxima[1]),t_i)
            # 2. Single Peak Detection with temporal threshold constraint
            if ((maxima[1][max(n_max,1)]-maxima[1][max(n_max-1,0)]) < Ki):
                index = maxima[0].index(min([maxima[0][max(n_max,1)],maxima[0][max(n_max-1,0)]]))     # Determine the index of the smallest peak
                maxima[0].pop(index)                                            # Delete smallest peak
                maxima[1].pop(index)   

        # Adaptive thresholds determination

        # Adaptive zero-velocity detection

        # Results
        print('Amount of peaks:', len(maxima[0]))
        print('Amount of valleys:', len(minima[0]))
        return maxima, minima

