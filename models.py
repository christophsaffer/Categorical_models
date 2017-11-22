import pandas as pd


class CategModel:

    def __init__(self, name):
        self.name = name
        self.path_data = 'data_sets/'
        self.path_models = 'models/'

#    def selection(dataset):

    def read_csv_data(self, nameoffile):
        data = pd.read_csv(self.path_data + nameoffile, index_col=0)
        for col in data.columns:
            data[col] = data[col].astype('category')

        return data

    def model_compl(self, data):
        compl = 1
        for col in data.columns:
            tempdata = data[col]
            cat = len(tempdata.values.categories)
            compl = compl * cat

        return compl

    def modelinfo(self, nameoffile):

        data = self.read_csv_data(nameoffile)
        compl = self.model_compl(data)

        print("Table shape: ", data.shape)
        print("Model Complexity: ", compl)
        print("Table head: ")
        print(data.head())

'''
    def marg(model):

    def cond(model):

    def aggr(model):

    def samp(model):

    def dens(model):
'''
