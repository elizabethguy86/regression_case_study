import pandas as pd
import numpy as np
from DummyMaker import DummyMaker
from sklearn.linear_model import LinearRegression

#df = pd.read_csv(filename, compression='zip') read in dataframe

'''Transform date column to pandas date-time format'''
def transform_date(df):
    df['saledate_dt'] = pd.to_datetime(df['saledate'])
    df['saleyear'] = df['saledate_dt'].apply(lambda x: x.year)
    df['sale_age'] = (df['saleyear']) - df['YearMade']
    #if year made is 1000, replace it with the sales year minus 10 years the sale age.
    df.loc[df['YearMade'] == 1000,'YearMade'] = df.loc[:,'saleyear'] - 10
    return df


def clean(df):
    #mean replace empty values
    df['MachineHoursCurrentMeter'].fillna(value = df['MachineHoursCurrentMeter'].mean(), inplace = True)
    #median replace empty values
    df['ProductSize'].fillna(value = 'Medium', inplace = True)
    df['Enclosure'].fillna(value = 'None or Unspecified', inplace = True)

    df = df.loc[:,['SalesID', 'sale_age', 'YearMade','MachineHoursCurrentMeter',
            'state', 'SalePrice', 'ProductSize', 'ProductGroupDesc',
            'Enclosure']]

    def region_category(string):
        '''Map individual states to geographic regions'''
        mapping = {1: ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont', 'New York', 'New Jersey',
                  'Pennsylvania', 'Washington DC'],
        2: ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin', 'Iowa',
            'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota'],
        3: ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia',
            'District of Columbia', 'West Virginia','Alabama', 'Kentucky', 'Mississippi', 'Tennessee', 'Arkansas',
            'Louisiana', 'Oklahoma', 'Texas'],
        4: ['Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Utah', 'Wyoming', 'Alaska',
            'California', 'Hawaii', 'Oregon', 'Washington'],
        5: ['Puerto Rico', 'Unspecified']}
        for key in mapping:
            if string in mapping[key]:
                return key

    df['regions'] = df['state'].apply(lambda x: region_category(x))

'''Make Dummy variables.  1 means column variable meets dummy criterion, else 0
if criterion not met'''

    dm = DummyMaker()
    dm.fit(df['regions'])
    region_dummies = dm.transform(df['regions'])

    dm.fit(df['ProductSize'])
    size_dummies = dm.transform(df['ProductSize'])

    dm.fit(df['ProductGroupDesc'])
    prod_dummies = dm.transform(df['ProductGroupDesc'])

    dm.fit(df['Enclosure'])
    encl_dummies = dm.transform(df['Enclosure'])

    df = df.reset_index()
    df.drop('index', axis = 1, inplace = True)
    merged = df.join(region_dummies).join(size_dummies).join(prod_dummies).join(encl_dummies)
    # merged = df.join(usage_dummies)

    merged = merged.drop(
        ['state','regions', 'ProductSize', 'ProductGroupDesc', 'Enclosure'], axis = 1)
    if 'NO ROPS' in merged.columns:
        merged = merged.drop('NO ROPS', axis = 1)

    X = merged.drop(['SalePrice', 'SalesID'], axis = 1)
    y = np.log(merged['SalePrice'])
    ids = merged['SalesID']

    return X, y, ids
