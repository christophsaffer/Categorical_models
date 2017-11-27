import pandas as pd


def read_csv_data(path_data, nameoffile):
    data = pd.read_csv(path_data + nameoffile, index_col=0)
    for col in data.columns:
        data[col] = data[col].astype('category')

    return data


def import_model(path_data, modelname):
    data = pd.read_csv(path_data + modelname, index_col=0)
    return data


def data_compl(data):
    compl = 1
    for col in data.columns:
        tempdata = data[col]
        cat = len(tempdata.values.categories)
        compl = compl * cat

    return compl