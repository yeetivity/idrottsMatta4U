import json
from matplotlib import pyplot as plt


class DataProcesser():
    """
    Class that can both load data from json files and export data into a nice plot
    """

    def __init__(self):
        """
        Initialising the class

        =INPUT=

        =OUTPUT=

        =NOTES=

        """
    
    def load_json(self, path):
        f = open(path)
        data = json.load(f)
        return data

    def plot_jsondata(self):
        """
        This will become the function to plot the data
        """



