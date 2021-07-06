import pandas as pd
import numpy as np
import scipy.io as spio

class DataReader:
    def __init__(self):
        pass

    @staticmethod
    def read_GL():
        data = {}
        for i in range(1,101):
            testData = spio.loadmat('genData_smG_rlG\sub_'+str(i)+'.mat')
            struct = testData['subData']
            data['S'+str(i)] = [
                        {
                            'action': np.array([struct[0,0]['resp'].flatten()]),
                            'state': np.array([struct[0,0]['stimOffers']]),
                            'reward': np.array([struct[0,0]['outcomeRew'].flatten()]),
                            'id': 'S'+str(i),

                            'block': 0
                        }

                    ]
        return data

    @staticmethod
    def read_GL_index():
        return pd.read_csv("../data/BD/ind.csv", header=0, sep=',', quotechar='"')

if __name__ == '__main__':
    DataReader.read_GL()
