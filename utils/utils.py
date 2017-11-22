import pandas as pd

def read_csv_data(path_data, nameoffile):
    data = pd.read_csv(path_data + nameoffile, index_col=0)
    for col in data.columns:
        data[col] = data[col].astype('category')

    return data
