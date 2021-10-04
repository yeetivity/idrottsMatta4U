import numpy as np

class DataProcess(object):
    """
    
    """
    def __init__(self, storeddata):
        self.storeddata = storeddata

        self.gravity = 9.81
        self.dT = 1/52
        self.combAcc = []
        self.simple_kalAcc = []
        self.kalAcc = []
        self.kalVel = []
        self.kalPos = []
        self.kalData = []
        self.gyroX = []
        for i in range(len(self.storeddata)):
            self.gyroX.append(self.storeddata[i]['gyrX'])

        # might not be necessary? see later
        self.accX = []
        for i in range(len(self.storeddata)):
            self.accX.append(self.storeddata[i]['accX'])
        self.accY = []
        for i in range(len(self.storeddata)):
            self.accY.append(self.storeddata[i]['accY'])
        self.accZ = []
        for i in range(len(self.storeddata)):
            self.accZ.append(self.storeddata[i]['accZ'])

        return

    
    def combineAccelerations(self):
        """
        Function that processes the accelerations
        """
        for i in range(len(self.storeddata)):
            self.combAcc.append(np.sqrt(np.square(self.storeddata[i]['accX']) + 
                                        np.square(self.storeddata[i]['accY']) +
                                        np.square(self.storeddata[i]['accZ']))
                                        - self.gravity )

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


    def complexKalmanFilter(self, data_to_filter, B=None, u=None):
        """
        Kalman with multiple dimensions
        """
        if (B is None) and (u is None):
            B = np.zeros((3,3))
            u = np.zeros((3,1))
        else:
            pass

        
        for i in range(len(data_to_filter)):
        # settings for acc kalman
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
            z = data_to_filter[i]
            x = np.array([  [0],
                            [0],
                            [0]])                       # Position, velocity, acceleration
            y = np.subtract( z[0], np.dot(H,x))         # Comparing predicted value with measurement
            X = x

            # filtering
            for j in range (len(z)):
                
                # PREDICTION VALUES
                    # x = Ax + Bu
                x = np.dot(A,x) + np.dot(B, u)
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


    def complexKalmanFilterGyro(self, B=None, u=None):
        """
        Kalman with multiple dimensions, for gyro
        """
        if (B is None) and (u is None):
            B = np.zeros((3,3))
            u = np.zeros((3,1))
        else:
            pass


        for i in range(len(data_to_filter)):
            # setting for gyro Kalman
            A = np.array([  [1, self.dT],     # angle
                        [0, 1]])    # angular v     # State transition matrix
            P = np.array([  [0, 0],
                        [0, 10]])                # State covariance matrix
            Q = np.array([  [0, 0],
                        [0, 0.5]])               # Process noise covariance matrix
            H = np.array([0, 1])                     # Measurement matrix

            #B = np.array([[]])
            u = np.array([[self.accX], #i won't work, not in a loop
                     [self.accY],
                     [self.accZ]])

            # Initiliaze some filter values
            R = 10                                      # Some scalar
            z = self.gyroX
            x = np.array([  [0],
                            [0]])                       # angle, angular v
            y = np.subtract( z[0], np.dot(H,x))         # Comparing predicted value with measurement
            X = x

            # filtering
            for j in range (len(z)):
                
                # PREDICTION VALUES
                    # x = Ax + Bu
                x = np.dot(A,x) + np.dot(B, u)
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
    
    
    def filter():
        """
        If you don't want to sleep, make EMWA filter
        """

        return