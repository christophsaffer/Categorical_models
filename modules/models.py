import pandas as pd
import itertools
import numpy as np

from utils.utils import *


class Categ:

    # Sets intial variables for an instance of the class
    def __init__(self, name):
        self.name = name
        self.path_data = 'data_sets/'
        self.path_models = 'models/'
        self.path_samples = 'samples/'
        self.data = 0
        self.model = 0

    # Returns some variables of the current model + its based data
    def model_info(self):

        model = self.model
        data = self.data
        compl = len(model)

        print("Data shape: ", data.shape)
        print("Model Complexity: ", compl)
        print("Model head: ")
        print(model.head())

    # Fits a model to given data
    def selection(self, save=True, data=None):

        # If no data is given, take the data of the class object
        if data is None:
            data = self.data

        # Get the complexity, in other words the length of the model 
        compl = data_compl(data)

        # Prepare modeltable
        # Generate a list of all possible vectors of the data
        values_cat = []
        for col in data.columns:
            values_cat.append(data[col].values.categories)
        values_cat.append([0])
        possible_vectors = list(itertools.product(*values_cat))

        # Generate a list of the columns of the model
        # Basically the same columns as the data + column 'p' for probability
        columns = list(data.columns)
        columns.append('p')
        modeltable = pd.DataFrame(columns=columns)

        # Initialize the model
        for i in range(0, compl):
            modeltable.loc[i] = possible_vectors[i]

        # Calculate parameters by counting the vectors in the data
        numb_columns = len(data.columns)
        numb_data = len(data)
        for i in range(0, compl):
            row = modeltable.iloc[i, 0:numb_columns]
            temp_table = data.isin(list(row)).T
            numb = ((temp_table == True).sum() == numb_columns).sum()
            modeltable.iloc[i, numb_columns] = float(numb) / float(numb_data)

        # Save the model in the model path of the object if the parameter save is true
        if save:
            modeltable.to_csv(self.path_models + 'model_' + self.name + '.csv')
            print("Saved model in: ", self.path_models)

        return modeltable

    # Generate sample points from the model of the object, k is number of sample points
    def sampling(self, k=100, save=True):

        model = self.model

        # Get the complexity of the model
        compl = len(model)

        # Initialize the table of the sample points
        columns = list(model.columns)
        columns.remove('p')
        samples = pd.DataFrame(columns=columns)

        # Initialize and generate list of buckets for the Inverse Transform Sampling
        buckets = pd.DataFrame(columns=['limits'])
        buckets.loc[0] = model.p.loc[0]
        for i in range(1, len(model)):
            buckets.loc[i] = model.p.loc[i] + buckets.loc[i - 1]

        # Initialize k random numbers between 0 and 1 and assign it to its bucket
        # Based on that generate the random vector and assign it in the sample list
        for i in range(0, k):
            rand = np.random.uniform()
            for j in range(0, len(model)):
                if rand <= float(buckets.loc[j]):
                    break
            samples.loc[i] = list(model.iloc[j, 0:len(columns)])

        # Save the columns as categories
        for col in samples.columns:
            samples[col] = samples[col].astype('category')

        if save:
            samples.to_csv(self.path_samples + 'sampling_' +
                           str(k) + '_' + self.name + '.csv')
            print("Saved samples in: ", self.path_samples)

        return samples

    # Return the argmax or a list of the argmax's of a model 
    def argmaximum(self, model=None):

        if model is None:
            model = self.model

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

        # Generate output
        perc = float(value) * 100
        print("OUTPUT############################")
        print("Score: ", value, " - Prob: ", str(perc), "%")
        print("Index/Indizes of Sol: ", ind)
        print("MAX at: ")
        print(solution)
        print("OUTPUT - END #####################")

        return solution

    # Returns the score of a given vector of the model
    def dens(self, values, model=None):

        if model is None:
            model = self.model

        model_without_dens = model.iloc[:, 0:len(model.columns) - 1]
        values = list(values)
        ind = model_without_dens.isin(values).all(axis=1)
        ind = pd.DataFrame(ind, columns=["Bool"])
        ind = list(ind.index[ind['Bool']])

        return model.iloc[ind[0], len(model.columns) - 1]

    # Calculates the mean square or absolute error between two given models
    def error(self, sampling, method="abs", model=None):

        if model is None:
            model = self.model

        if method == "abs":
            return (abs(model.p - sampling.p)).sum()
        if method == "square":
            return ((model.p - sampling.p)**2).sum()

    # Marginalize out one column (based on its name OR index) of a model
    def marg(self, margout, model=None):

        if model is None:
            model = self.model

        # Check if paramter is given as string or index (int)
        if type(margout) != str:
            margout = model.columns[margout]

        # Generates the columns of the new model
        columns = list(model.columns)
        columns.remove(margout)

        # Generates a table of the new model without the column 'p'
        margmod = model.loc[:, columns]
        no_dens = margmod.iloc[:, 0:len(margmod.columns) - 1]

        # Generates the final model that will be returned
        margmod_final = pd.DataFrame(columns=columns)

        # Process of marginalization
        k = 0
        for i in range(0, len(model)):
            if i in no_dens.index:
                ind = no_dens.isin(list(no_dens.loc[i])).all(axis=1)
                ind = pd.DataFrame(ind, columns=["Bool"])
                ind = list(ind.index[ind['Bool']])
                prob = 0
                for j in ind:
                    prob = prob + margmod.p[j]
                newrow = list(no_dens.loc[i])
                newrow.append(prob)

                margmod_final.loc[k] = newrow
                k = k + 1

                no_dens = no_dens.drop(ind)

        return margmod_final

    # Margilize over a list of given columns (name or index of the columns)
    def marg_list(self, margout):

        model = self.model

        margout.sort(reverse=True)
        margmod = model
        for mod in margout:
            margmod = self.marg(mod, margmod)

        return margmod

    # Conditionalize over a column and given value
    def cond(self, col, value, model=None):

        if model is None:
            model = self.model

        # Checks if parameter is given as string or index (int)
        if type(col) != str:
            col = model.columns[col]

        # Checks if column should be assigned to only one or more values
        if type(value) == str:
            temp = value
            value = []
            value.append(temp)

        # Removes the rows where the column is not one of the values
        condmodtemp = model.loc[:, col].isin(value)
        condmodtemp = model.loc[condmodtemp]

        # Calculates the denominator for calculating the conditional distributions 
        denominator = condmodtemp.p.sum()

        # Initialize the conditional model
        columns = list(model.columns)
        condmod_final = pd.DataFrame(columns=columns)
        columns.remove("p")

        # Process of Conditionalization
        k = 0
        for i in condmodtemp.index:
            numerator = model.p.loc[i]
            newrow = list(condmodtemp.loc[i, columns])
            p = numerator / denominator
            newrow.append(p)

            condmod_final.loc[k] = newrow
            k = k + 1

        return condmod_final
