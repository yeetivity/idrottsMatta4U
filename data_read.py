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

    def load_all(self, paths):
        for i in range(len(paths)):
            path = 'JSON_DATA/%s' % paths[i]
            f = open(path)
            self.data.append(json.load(f))

        return self.data

    