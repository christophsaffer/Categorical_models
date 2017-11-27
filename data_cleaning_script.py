import pandas as pd
temp_table = pd.read_stata('data_sets/allbus2016/allbus2016.dta',
                           columns=['educ', 'di01a', 'eastwest', 'sex', 'age', 'pa02a', 'pv01'])
data = pd.DataFrame(temp_table)

data.columns = ["a_education", "b_income", "c_eastwest", "d_sex",
                "e_age", "f_political_interest", "g_preferred_party"]

dic_columns = {"a_education": [], "b_income": [], "c_eastwest": [], "d_sex": [
], "e_age": [], "f_political_interest": [], "g_preferred_party": []}

cleaned_data = pd.DataFrame(dic_columns, dtype='category')

print(data.g_preferred_party.values.categories)

def change_entry(entry, j):
    if j == 0:
        if entry in ['DATENFEHLER', 'KEINE ANGABE', 'NOCH SCHUELER', 'ANDERER ABSCHLUSS']:
            return 0
        if entry in ['OHNE ABSCHLUSS', 'VOLKS-,HAUPTSCHULE']:
            return 'EHER UNGEBILDET'
#        if entry in ['MITTLERE REIFE']:
#            return 'MITTEL'
        if entry in ['MITTLERE REIFE', 'FACHHOCHSCHULREIFE', 'HOCHSCHULREIFE']:
            return 'EHER GEBILDET'
    if j == 1:
        if entry in ['DATENFEHLER', 'VERWEIGERT']:
            return 0
        if entry in ['KEIN EINKOMMEN']:
            return 'EHER NIEDRIG'
#        if entry < 1000:
#            return 'NIEDRIG'
        if entry < 2500:
            return 'EHER NIEDRIG'
        if entry >= 2500:
            return 'EHER HOCH'
    if j == 2:
        return entry
    if j == 3:
        return entry
    if j == 4:
        if entry == 'NICHT GENERIERBAR':
            return 0
        if entry < 45:
            return 'JUNG'
        if entry > 44:
            return 'ALT'
    if j == 5:
        if entry in ['SEHR STARK', 'STARK', 'MITTEL']:
            return 'JA'
        if entry in ['WENIG', 'UEBERHAUPT NICHT']:
            return 'NEIN'
    if j == 6:
        if entry in ['NICHT WAHLBERECHTIGT', 'DATENFEHLER', 'KEINE ANGABE', 'WEISS NICHT', 'VERWEIGERT', 'ANDERE PARTEI', 'PIRATEN', 'NPD']:
            return 0
        else:
            return entry


i = 0
k = 0
while i < data.count()[0]:
    j = 0
    current_row = []

    while j < 7:
        entry = data.loc[i][j]
        value = change_entry(entry, j)
        current_row.append(value)
        j = j + 1

    # print(current_row)
    if 0 not in current_row:
        cleaned_data.loc[k] = current_row
        k = k + 1

    i = i + 1
    print("i: ", i, ", k: ", k)

for col in cleaned_data.columns:
    cleaned_data[col] = cleaned_data[col].astype('category')

print(cleaned_data.head())

cleaned_data.to_csv('dataset.csv')

print(cleaned_data.g_preferred_party.values.categories)
