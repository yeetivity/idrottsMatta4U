import json
import csv
import sys
import glob
import numpy as np
from matplotlib import pyplot as plt


class DataRead():
    """
    Class that can both loads data from json files
    """

    def __init__(self):
        """
        Initialising the class
        """

        self.data = []
        self.transdata = []
        return 


    def read(self, folderpath, filetype='csv'):
        if (filetype == 'csv'):
            p = sorted(glob.glob(folderpath + '/ACC' + "/*.csv"))
            data_acc = self.load_csvs(p)
            self.transform_csvformat(data_acc, type='acc')

            p = sorted(glob.glob(folderpath + '/GYRO' + "/*.csv"))
            data_gyr = self.load_csvs(p)
            data = self.transform_csvformat(data_gyr, type='gyro')

        elif (filetype == 'json'):
            p = sorted(glob.glob(folderpath + "/*.json"))
            data = self.load_jsons(p)

        else:
            sys.exit('Wrong filetype is given')
        
        return data

    def load_jsons(self, paths):
        """
        Function that reads a folder of JSON files

        =INPUT=
            self        Datastructure to save the read data in
            paths       array with relative paths to the seperate files
        =OUTPUT=
            self.data   Datastructure with all the data
        """
        
        for i in range(len(paths)):
            f = open(paths[i])
            self.data.append(json.load(f))

        return self.data

   
    def load_csvs(self, paths):
        """
        Function that reads a folder of CSV files

        =INPUT=
            self        Datastructure to save the read data in
            paths       array with relative paths to the seperate files
        =OUTPUT=
            self.data   Datastructure with all the data
        """
        self.csvdata = []
        for i in range(len(paths)):
            with open(paths[i], newline='') as csvfile:
                self.csvdata.append(list(csv.reader(csvfile)))

        return self.csvdata


    def transform_csvformat(self, data_container, type='acc'):
        """
        Function that transforms the sorting of the data output

        =INPUT=
            self                Datastructure to save the read data in
            data_container      Data_container that needs to be transformed
        =OUTPUT=
            self.transdata      Datastructure with all the transformed data
        """

        for i in range(len(data_container)):
            # Transforming data
            if (type == 'acc'):             # Fill in the acceleration data
                self.transdata.append({'accX': [], 'accY': [], 'accZ': [], 'gyrX': [], 'gyrY': [], 'gyrZ': [], 'time_a': [], 'time_g': []})
                for ii in range(len(data_container[i])):
                    if (ii != 0):
                        self.transdata[i]['time_a'].append(float(data_container[i][ii][0]))
                        self.transdata[i]['accX'].append(float(data_container[i][ii][1]))
                        self.transdata[i]['accY'].append(float(data_container[i][ii][2]))
                        self.transdata[i]['accZ'].append(float(data_container[i][ii][3]))
            else:                           # Fill in the gyro data
                for jj in range(len(data_container[i])):
                    if (jj != 0):
                        self.transdata[i]['time_g'].append(float(data_container[i][jj][0]))
                        self.transdata[i]['gyrX'].append(float(data_container[i][jj][1]))
                        self.transdata[i]['gyrY'].append(float(data_container[i][jj][2]))
                        self.transdata[i]['gyrZ'].append(float(data_container[i][jj][3]))
            
            # Time check
            if (self.transdata[i]['time_a'] == self.transdata[i]['time_g']):
                if ((self.transdata[i]['time_a'] != 0) and (self.transdata[i]['time_g'] != 0)):     #transform begin time to zero
                    self.transdata[i]['time_a'] = np.array(self.transdata[i]['time_a']) - self.transdata[i]['time_a'][0]
                    self.transdata[i]['time_g'] = np.array(self.transdata[i]['time_g']) - self.transdata[i]['time_g'][0]
            elif ((self.transdata[i]['time_g'] == []) or (self.transdata[i]['time_a'] == [])):      #If one of the times is still empty skip the loop
                pass
            else:
                sys.exit("Timestamps are not the same")

        return self.transdata