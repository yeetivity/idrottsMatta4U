import numpy as np
from settings import Settings as s
import sys

class KalmanFilter(object):
    """
    """

    def __init__(self, data, control_data=None, Type='Acc'):
        """
        =INPUT=
            Data:   data from 1 experiment
        """
        self.data = data
        self.control_data = control_data        
        self.filtered_data = []
        self.type = Type

        return

    def computeAttitudes(self, direction='x'):
        # Todo : see how this should function
        x = np.array(self.control_data[0])
        y = np.array(self.control_data[1])
        z = np.array(self.control_data[2])

        if direction == 'x':
            pitch = np.tan(x / (np.sqrt(y**2 + z**2)))
            return pitch
        elif direction == 'y':
            roll = np.tan(y/ (np.sqrt(x**2 + z**2)))
            return roll
        elif direction == 'z':
            yaw = np.tan((np.sqrt(x**2 + y**2))/z)
            return yaw

    def kalmanFilter(self, reset_times=None, direction='x'):
        """
        =OUTPUT=
        X ->    If the filter is done for gyro data the shape is [2,x] 
                with [0] is angle, [1] angular velocity
        """
        #* Initialisation
        # Initialise the pitch yaw and roll based on control data
        if self.control_data != None:
           cntrl = self.computeAttitudes(direction=direction)

        if (self.type == 'Acc'):
            A, P, Q, H, R, x, X, y = self.reset3x3()
            for j in range(len(self.data)):
                x, X, P = self.runFilter(A, x, X, P, Q, H, j, R, z=self.data)
            X = X[:,1:] #Delete first column
            return X[0], X[1], X[2]

        elif (self.type == 'Gyro'):
            A, P, Q, H, B, R, x, X, y = self.reset2x2(control=cntrl)
            for j in range(len(self.data)):
                x, X, P = self.runFilter(A, x, X, P, Q, H, j, R, B, u=cntrl[j], z=self.data)
            X = X[:,1:] #Delete first column
            return X

        else:
            sys.exit('Wrong type is given')


    def runFilter(self, A, x, X, P, Q, H, index, R, B=None, u=None, z=None):
        # PREDICTION VALUES
        if u is not None:
            u = np.array(   [[0], [u]])
            Bu = np.dot(B, u)
            Ax = np.dot(A, x)
            for i in range(2):
                x[i] = Ax[i] + Bu[i]
        else:
            x = np.dot(A,x)

        P = np.add((np.dot(np.dot(A,P), A.transpose())), Q)
        
        # MEASUREMENT VALUES
        y = np.subtract(z[index], np.dot(H, x))
        K = np.dot( P, H.transpose()) / (np.dot(np.dot(H, P), H.transpose()) + R)

        # UPDATE X AND P
        if self.type == 'Acc':
            for i in range(0,3):
                x[i] = x[i] + y*K[i]
        elif self.type == 'Gyro':
            for i in range(0,2):
                x[i] = x[i] + y*K[i]

        P = np.dot(np.subtract(1, np.dot(K,H)), P)
        X = np.hstack((X, x))

        return x, X, P


    def reset3x3(self, index=0, X=None):
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

        y = np.subtract( self.data[index], np.dot(H, x))         # Comparing predicted value with measurement
        return A, P, Q, H, R, x, X, y


    def reset2x2(self, control, index=0):
        #* Initialisation
        A = np.array([  [1, s.f_sampling],       # angle
                        [0, 1]])                 # State transition matrix
        P = np.array([  [0, 0],
                        [0, 10]])                # State covariance matrix
        Q = np.array([  [0, 0],
                        [0, 0.5]])               # Process noise covariance matrix
        H = np.array(   [0, 1])                  # Measurement matrix

        B = np.array([  [0, 0],
                        [0, 1]])                #Todo: tune

        # Initiliaze some filter values
        R = 10                                      # Some scalar
        x = np.array([  [0],
                        [0]])                       # angle, angular v
        y = np.subtract(self.data[index], np.dot(H,x))         # Comparing predicted value with measurement
        # u = np.array(   [[0],
        #                 [control[index]]])
        X = x

        return A, P, Q, H, B, R, x, X, y