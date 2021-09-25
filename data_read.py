import json
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
        return

    def load_all(self, folderpath, paths):
        for i in range(len(paths)):
            path = folderpath+paths[i]
            f = open(path)
            self.data.append(json.load(f))

        return self.data

    