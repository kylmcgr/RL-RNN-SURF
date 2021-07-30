import pandas as pd

class DataReader:
    def __init__(self):
        pass

    @staticmethod
    def read_EE():
        data = pd.DataFrame()
        for i in range(1,101):
            new_data = pd.read_csv('../data/EE/simulationData_beta10_halfhalf/sim_'+str(i)+'.csv', header=0, sep=',', quotechar='"')
            new_data['id'] = new_data['subID']
            new_data['block'] = new_data['blockID']
            new_data['reward'] = new_data['outcome']
            new_data['action'] = [x-1 for x in new_data['respKey']]
            new_data['diag'] = ['Novelty' if i <= 50 else 'Uncertainty' for x in new_data.index]
            del new_data['subID']
            del new_data['blockID']
            del new_data['outcome']
            del new_data['respKey']
            data = data.append(new_data, ignore_index=True)
        return data

    @staticmethod
    def read_EE_index():
        return pd.read_csv("../data/EE/ind.csv", header=0, sep=',', quotechar='"')

if __name__ == '__main__':
    DataReader.read_EE()
