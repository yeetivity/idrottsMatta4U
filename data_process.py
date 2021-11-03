import numpy as np
from settings import Settings as s

class DataProcess(object):
    """
    
    """
    def __init__(self, storeddata): #TODO clean up the init
        self.storeddata = storeddata
        self.gyro = [[],[],[]]
        self.acc = [[],[],[]]
        self.time = storeddata[s.experiment]['time_a'] 

        self.combAcc = []
        self.kalData = []

        maxima = []

        # Fill in the storeddata
        for i in range(len(self.storeddata)):
            self.acc[0].append(self.storeddata[i]['accX'])
            self.acc[1].append(self.storeddata[i]['accY'])
            self.acc[2].append(self.storeddata[i]['accZ'])
            self.gyro[0].append(self.storeddata[i]['gyrX'])
            self.gyro[1].append(self.storeddata[i]['gyrY'])
            self.gyro[2].append(self.storeddata[i]['gyrZ'])
            maxima.append(len(self.storeddata[i]['time_a']))
        
        maximum = max(maxima)
        self.pitch = np.zeros((len(self.storeddata), maximum))
        self.roll = np.zeros((len(self.storeddata), maximum))
        self.yaw = np.zeros((len(self.storeddata), maximum))
        self.accX_arr = np.zeros((len(self.storeddata), maximum))       #! Can't we make everything an array?
        self.accY_arr = np.zeros((len(self.storeddata), maximum))
        self.accZ_arr = np.zeros((len(self.storeddata), maximum))

        # Check if we want this in initialisation. 
        for i in range(len(self.storeddata)):
            for j in range(len(self.acc[0][i])):
                self.accX_arr[i][j] = self.acc[0][i][j]
                self.accY_arr[i][j] = self.acc[1][i][j]
                self.accZ_arr[i][j] = self.acc[2][i][j]
                self.pitch[i][j] = np.tan(self.accX_arr[i][j]/ (np.sqrt(self.accY_arr[i][j]**2 + self.accZ_arr[i][j]**2)))
                self.roll[i][j] = np.tan(self.accY_arr[i][j]/ (np.sqrt(self.accX_arr[i][j]**2 + self.accZ_arr[i][j]**2)))
                self.yaw[i][j] = np.tan((np.sqrt(self.accX_arr[i][j]**2 + self.accY_arr[i][j]**2))/self.accZ_arr[i][j])

        return

    
    def combineAccelerations(self):
        """
        Function that combines the accelerations
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

        acc_simpKal = []
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

            acc_simpKal.append(x)
        return acc_simpKal


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

        Todo Check if self.kalData needs to be global or local (using overwriting and such)
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
                y = np.subtract( z[0], np.dot(H,x))         # don't think this actually does something

                # Filtering
                # Todo: See if we can put code line 135-150 in a function (since elif also uses it)
                for j in range (len(z)):
                    x, P, K, P, X = self.runFilter(A,x,X,P,Q,z,H,j,R)

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

                x, P, K, P, X = self.runFilter(A,x,X,P,Q,z,H,j,R)
            
            X = X[:,1:]

            self.kalData = X

        return self.kalData


    def runFilter(self, A, x, X, P, Q, z, H, j, R):
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

        return x, P, K, P, X


    def resetKalman(self, z, index, X=None):
        """
        Todo: Add documentation
        """
        A = np.array([  [1, s.f_sampling, 0.5 * s.f_sampling**2],
                        [0, 1, s.f_sampling],
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
        
        TODO: can't this function be merged with the other complexKalmanFilter?

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
            A = np.array([  [1, s.f_sampling],     # angle
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


    def horizontalComponent(self, angle):
        """
        Compute the horizontal component of acc (or vel)
        #Todo: check if right axes are used.
        =INPUT=
            angle: angles given out by kalman filter on gyro (6x2x1 array:
                        [experiment][angle=0 (degrees), angular velocity = 1][timestamp])

        =OUTPUT=
            self.horComp    6x1024 array with:
                            [experiment][timestamp]
        """

        # init list with horizontal components
        acc_fixed_coord = np.zeros((3,1))  #! Why is this in self? Does another function need it?
        rotation_matrix = np.zeros((3,3))       #! Not sure if we need this initialisation step
        
        for i in range(len(angle)): #experiments

            #takes one row of self.acc[2] (one experiment)
            acc_sensor_coord = np.array(    [self.acc[0][i],
                                            self.acc[1][i],
                                            self.acc[2][i]])

            # all the values in one experiment
            for j in range(len(acc_sensor_coord[0])):
                #! np.cos takes radians 
                # Todo: check if kalmangyro outputs radians
                rotation_matrix = np.array([    [np.cos(angle[i][0][j]),    0,  -np.sin(angle[i][0][j])],
                                                [0,                         1,     0],
                                                [np.sin(angle[i][0][j]),    0,  np.cos(angle[i][0][j])]])
                acc_fixed_coord = np.dot(rotation_matrix, acc_sensor_coord[:,[j]]) #np.cos(angle[i][0][j])) #[0]=angle, [1]=angular velocity
                #! Self.acc_fixed_coord is being overwritten in every loop. Don't know if that is what you want?
            
            
        return acc_fixed_coord #needs to be changed
    
    
    def emwaFilter(self,data,alpha):
        """
        EMWA filter
        """
        #Initialization
        emwaData = [data[0]]

        #Filtering
        for k in range(1, len(data)):
            emwaData.append(alpha*self.emwaData[k-1]+(1-alpha)*data[k])
            
        return emwaData

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
        indices = []

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

