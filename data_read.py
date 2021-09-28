import json
import csv
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
        self.csvdata = []

        return

    def load_json(self, path):
        """
        Funtion that reads a single JSON file
        """
        f = open(path)
        data = json.load(f)

        return data

    def load_jsons(self, folderpath, paths):
        """
        Function that reads a folder of JSON files

        =INPUT=
            self        Datastructure to save the read data in
            folderpath  Path to the folder that should be read
            paths       array with relative paths to the seperate files
        =OUTPUT=
            self.data   Datastructure with all the data
        """
        
        for i in range(len(paths)):
            path = folderpath+paths[i]
            f = open(path)
            self.data.append(json.load(f))

        return self.data

    def load_csv(self, path):
        """
        Function that reads a single CSV
        """
        with open(path, newline='') as csvfile:
            data = list(csv.reader(csvfile))

        return data
    
    def load_csvs(self, folderpath, paths):
        """
        Function that reads a folder of CSV files

        =INPUT=
            self        Datastructure to save the read data in
            folderpath  Path to the folder that should be read
            paths       array with relative paths to the seperate files
        =OUTPUT=
            self.data   Datastructure with all the data
        """

        for i in range(len(paths)):
            path = folderpath+paths[i]
            with open(path, newline='') as csvfile:
                self.csvdata.append(list(csv.reader(csvfile)))

        return self.csvdata