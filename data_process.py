import numpy as np
from settings import Settings as s

class DataProcess(object):
    """
    
    """
    def __init__(self, storeddata):
        self.storeddata = storeddata

        self.combAcc = []
        self.simple_kalAcc = []
        self.kalAcc = []
        self.kalVel = []
        self.kalPos = []
        self.kalData = []
        self.time = storeddata[s.experiment]['time_a']
        
        self.emwaData = []
        
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


    def complexKalmanFilter(self):
        """
        Kalman with multiple dimensions
        """
        for i in range(len(self.combAcc)):
            # Initiliaze some filter values
            dT = 1/52
            R = 10                                      # Some scalar
            z = self.combAcc[i]
            A = np.array([  [1, dT, 0.5 * dT**2],
                            [0, 1, dT],
                            [0, 0, 1]])                 # State transition matrix
            P = np.array([  [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 10]])                # State covariance matrix
            Q = np.array([  [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0.5]])               # Process noise covariance matrix
            H = np.array([0, 0, 1])                     # Measurement matrix
            x = np.array([  [0],
                            [0],
                            [0]])                       # Position, velocity, acceleration
            y = np.subtract( z[0], np.dot(H,x))         # Comparing predicted value with measurement
            X = x

            # filtering
            for j in range (len(z)):
                
                # PREDICTION VALUES
                    # x = Ax + Bu
                x = np.dot(A,x) #+ np.dot(B, u)
                    # P = A P A^T + Q
                P = np.add( (np.dot(np.dot(A,P), A.transpose())), Q)
                
                # MEASUREMENT VALUES
                    # Y = Z - H X
                y = np.subtract(z[j], np.dot(H, x))
                    # K = (P H^T) / ( ( HPH^T) + R)
                K = np.dot( P, H.transpose()) / (np.dot(np.dot(H, P), H.transpose()) + R)

                # UPDATE X AND P
                    # X = X + KY
                for ii in range(0,3):
                    x[ii] = x[ii] + y*K[ii]
                    # P = (1 - KH) P
                P = np.dot(np.subtract(1, np.dot(K,H)), P)
                
                X = np.hstack((X, x))
            X = X[:,1:]
            self.kalData.append(X)


        return self.kalData
    
    
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
        K0 = 350        # Initial time interval threshold of Ki
        Ki = K0
        alpha = 0.7     # Scale factor used to determine the time interval threshold
        W2 = 5          # Number of consecutive valleys
        TH_pk = 40      # Peak detection threshold to exclude false detection
        TH_s = 190      # Fixed value to detect static states and determine whether to stop the update of K_i
        
        
        TH = 6          # Statistical value that used to distinguish the state of motion is intense or gentle  
        W1 = 3          # The window size of the acceleration-magnitude detector
        TH_vy = 1.9     # Valley detection threshold that utilized to detect the valleys 

        maxima = [[],[]]        # Array with maxima
        minima = [[],[]]        # Array with minima


        #* Valid valley detection
        # 1. Minima detection
        for i in range(1,len(combAcc)-1):
            if ((combAcc[i] < combAcc[i+1]) and (combAcc[i] < combAcc[i-1]) and (combAcc[i] < TH_vy)):
                minima[0].append(combAcc[i])
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
        for i in range(1,len(combAcc)-1):
            if ((combAcc[i] > combAcc[i+1]) and (combAcc[i] > combAcc[i-1]) and (combAcc[i] > TH_pk)):
                maxima[0].append(combAcc[i])
                maxima[1].append(self.time[i])

        # 2. Single Peak Detection with temporal threshold constraint
        j = 1
        while j < len(maxima[0]):
            if ((maxima[1][j]-maxima[1][j-1]) < Ki):
                index = maxima[0].index(min([maxima[0][j],maxima[0][j-1]]))     # Determine the index of the smallest peak
                maxima[0].pop(index)                                            # Delete smallest peak
                maxima[1].pop(index)
                j = j
            else:
                j+= 1

        # Adaptive thresholds determination

        # Adaptive zero-velocity detection

        # Results
        print('Amount of peaks:', len(maxima[0]))
        print('Amount of valleys:', len(minima[0]))
        return maxima, minima