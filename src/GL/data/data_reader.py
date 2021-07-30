import pandas as pd
import numpy as np
import scipy.io as spio
import csv

class DataReader:
    def __init__(self):
        pass

    @staticmethod
    def read_GL():
        data = {}
        for i in range(1,101):
            with open('simulationData\sim_'+str(i)+'.csv', 'r' ) as theFile:
                reader = csv.reader(theFile)
                headers = next(reader, None)
                actions = []
                states = []
                rewards = []
                for line in reader:
                    actions.append(int(line[3]))
                    states.append([int(line[8]),int(line[9])])
                    rewards.append(int(line[5]))
                block = [
                        {
                            'action': np.array([actions])-1,
                            'state': np.array([states]),
                            'reward': np.array([rewards]),
                            'id': 'S'+str(i),

                            'block': 0
                        }

                    ]
                data['S'+str(i)] = block
        return data

    @staticmethod
    def read_GL_index():
        return pd.read_csv("../data/GL/ind.csv", header=0, sep=',', quotechar='"')

if __name__ == '__main__':
    DataReader.read_GL()
