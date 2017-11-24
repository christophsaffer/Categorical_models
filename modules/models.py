import pandas as pd
import itertools
import numpy as np

from utils.utils import *


class Categ:

    def __init__(self, nameoffile):
        self.nameoffile = nameoffile
        self.path_data = 'data_sets/'
        self.path_models = 'models/'
        self.path_samples = 'samples/'
        self.data = read_csv_data(self.path_data, self.nameoffile)

    def model_compl(self):
        data = self.data
        compl = 1
        for col in data.columns:
            tempdata = data[col]
            cat = len(tempdata.values.categories)
            compl = compl * cat

        return compl

    def modelinfo(self):

        data = self.data
        compl = self.model_compl(data)

        print("Table shape: ", data.shape)
        print("Model Complexity: ", compl)
        print("Table head: ")
        print(data.head())

    def selection(self):

        data = self.data
        compl = self.model_compl()

        # Prepare modeltable
        values_cat = []
        for col in data.columns:
            values_cat.append(data[col].values.categories)
        values_cat.append([0])
        possible_vectors = list(itertools.product(*values_cat))

        columns = list(data.columns)
        columns.append('p')
        modeltable = pd.DataFrame(columns=columns)

        for i in range(0, compl):
            modeltable.loc[i] = possible_vectors[i]

        # Calculate parameters
        numb_columns = len(data.columns)
        numb_data = len(data)
        for i in range(0, compl):
            row = modeltable.iloc[i, 0:numb_columns]
            temp_table = data.isin(list(row)).T
            numb = ((temp_table == True).sum() == numb_columns).sum()
            modeltable.iloc[i, numb_columns] = float(numb) / float(numb_data)

        modeltable.to_csv(self.path_models + 'model_' + self.nameoffile)
        print("Saved model in: ", self.path_models)

        return modeltable

    def sampling(self, model, k):

        compl = self.model_compl()
        columns = list(self.data.columns)
        samples = pd.DataFrame(columns=columns)

        buckets = pd.DataFrame(columns=['limits'])

        buckets.loc[0] = model.p.loc[0]
        for i in range(1, len(model)):
            buckets.loc[i] = model.p.loc[i] + buckets.loc[i - 1]

        for i in range(0, k):
            rand = np.random.uniform()
            for j in range(0, len(model)):
                if rand <= float(buckets.loc[j]):
                    break
            samples.loc[i] = list(model.iloc[j, 0:len(columns)])

        samples.to_csv(self.path_samples + 'sampling_' +
                       str(k) + '_' + self.nameoffile)
        print("Saved samples in: ", self.path_models)

        return samples

    def argmaximum(self, model):

        cond = True
        i = 0
        sorted_model = model.p.sort_values(ascending=False)
        solution = []
        ind = []
        while(cond):
            value = sorted_model.iloc[i]
            index = sorted_model.index[i]
            entries = model.iloc[index, 0:len(model.columns) - 1]

            if (value != sorted_model.iloc[i + 1]):
                cond = False

            solution.append(entries)
            ind.append(index)
            i = i + 1

        perc = float(value) * 100
        print("OUTPUT############################")
        print("Score: ", value, " - Prob: ", str(perc), "%")
        print("Index/Indizes of Sol: ", ind)
        print("MAX at: ")
        print(solution)
        print("OUTPUT - END #####################")

        return solution

    def dens(self, model, values):

        model_without_dens = model.iloc[:, 0:len(model.columns) - 1]
        values = list(values)
        ind = model_without_dens.isin(values)
        ind = ind[ind[ind.columns] == True].T.sum().idxmax()

        return model.iloc[ind, len(model.columns) - 1]


'''
    def marg(model):

    def cond(model):

    def dens(model):
'''
