import json
import csv
import numpy as np
from matplotlib import pyplot as plt


class DataRead():
    """
    Class that can both loads data from json files
    """

    def __init__(self):
        """
        Initialising the class

        =INPUT=

        =OUTPUT=

        =NOTES=

        """
        self.data = []
        self.transformeddata = []

        return

    def load_json(self, path):
        """
        Funtion that reads a single JSON file
        """
        f = open(path)
        data = json.load(f)

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


    def load_csv(self, path):
        """
        Function that reads a single CSV
        """
        with open(path, newline='') as csvfile:
            data = list(csv.reader(csvfile))

        return data

    
    def load_csvs(self, paths):
        """
        Function that reads a folder of CSV files

        =INPUT=
            self        Datastructure to save the read data in
            folderpath  Path to the folder that should be read
            paths       array with relative paths to the seperate files
        =OUTPUT=
            self.data   Datastructure with all the data
        """
        self.csvdata = []
        for i in range(len(paths)):
            with open(paths[i], newline='') as csvfile:
                self.csvdata.append(list(csv.reader(csvfile)))

        return self.csvdata


    def transform_csvformat(self, data_container, acc = True):

        # fill in the array
        i = 0
        for i in range(len(data_container)):
            if (acc == True):
                self.transformeddata.append({'accX': [], 'accY': [], 'accZ': [], 'gyrX': [], 'gyrY': [], 'gyrZ': [], 'time_a': [], 'time_g': []})
                for ii in range(len(data_container[i])):
                    if (ii != 0):
                        self.transformeddata[i]['time_a'].append(float(data_container[i][ii][0]))
                        self.transformeddata[i]['accX'].append(float(data_container[i][ii][1]))
                        self.transformeddata[i]['accY'].append(float(data_container[i][ii][2]))
                        self.transformeddata[i]['accZ'].append(float(data_container[i][ii][3]))

            else:
                for jj in range(len(data_container[i])):
                    if (jj != 0):
                        self.transformeddata[i]['time_g'].append(float(data_container[i][jj][0]))
                        self.transformeddata[i]['gyrX'].append(float(data_container[i][jj][1]))
                        self.transformeddata[i]['gyrY'].append(float(data_container[i][jj][2]))
                        self.transformeddata[i]['gyrZ'].append(float(data_container[i][jj][3]))

        return self.transformeddata