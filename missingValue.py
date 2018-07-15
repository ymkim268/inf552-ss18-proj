import numpy as np
import pandas as pd
import missingno as msno
import matplotlib as plt
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class_feature = ['airconditioningtypeid', 'architecturalstyletypeid','buildingclasstypeid','buildingqualitytypeid',
                'decktypeid','fips','hashottuborspa','heatingorsystemtypeid','pooltypeid10', 'pooltypeid2', 'pooltypeid7',
                'propertycountylandusecode','propertylandusetypeid','propertyzoningdesc','rawcensustractandblock',
                'regionidcity','regionidcounty', 'regionidneighborhood', 'regionidzip','storytypeid','typeconstructiontypeid',
                'fireplaceflag','taxdelinquencyflag','taxdelinquencyyear','censustractandblock']

quantity_feature = ['basementsqft','finishedfloor1squarefeet',
                    'calculatedfinishedsquarefeet','finishedsquarefeet12', 'finishedsquarefeet13',
                    'finishedsquarefeet15','finishedsquarefeet50', 'finishedsquarefeet6',
                    'garagetotalsqft','latitude', 'longitude','lotsizesquarefeet', 'poolsizesum',
                    'yardbuildingsqft17', 'yardbuildingsqft26','structuretaxvaluedollarcnt',
                    'taxvaluedollarcnt', 'landtaxvaluedollarcnt','taxamount']

integer_feature = ['bathroomcnt', 'bedroomcnt','calculatedbathnbr','fireplacecnt','numberofstories','fullbathcnt',
                   'garagecarcnt','poolcnt','unitcnt','roomcnt', 'threequarterbathnbr','yearbuilt',
                   'assessmentyear']

string_feature= ['hashottuborspa','propertycountylandusecode','propertyzoningdesc','taxdelinquencyflag', 'fireplaceflag']


# visualize missing value matrix
def visualize_missing_value(data):
    file = data
    msno.matrix(df=file, figsize=(15, 6), labels=True)


# visualize properties of missing value using barplot
def properties_of_missing_value(data):
    msno.bar(data, figsize=(20, 8), color="#34495e", fontsize=12, labels=True)


# label categorical features ['a','b','c'] -- [0,1,2]
def label_encode_features(data, features, show_info=False):
    """

    :type data: pandas DataFrame
    """
    df = data.loc[:, features]
    print('encoding labels ....')
    for index1, ele in enumerate(features):
        p = data.loc[:, ele]
        p = p.sort_values(ascending=True)
        size_mapping = dict()
        for index2, element in enumerate(p.unique()):
            size_mapping[element] = index2
        p = p.map(size_mapping)
        if show_info == True:
            print(ele)
        p = p.replace(len(p.unique()) - 1, np.nan)
        df.loc[:, ele] = p
    print('merging results ..... ')
    for ele in features:
        if show_info == True:
            print(ele)
        data[ele] = df.loc[:, ele]
    print('successfully done.....')
    return data


# Onehot encode categorical features ['a','b','c'] = [[1,-1,-1],[-1,1,-1],[-1,-1,1]]
def onehot_encode(data_feature_values, show_info=False):
    enc = OneHotEncoder()
    enc.fit(data_feature_values)
    data_feature_values = enc.transform(data_feature_values)
    print("after encoding:")
    print(data_feature_values.shape)
    if show_info == True:
        print(enc.n_values_)
    return data_feature_values


# print the number of missing values of each features
def print_missing(data):
    print('Each feature missing value:')
    print(data.isnull().sum().sort_values())


# fill data set missing value with mean
def fill_with_mean(data, features, show_info=False):
    """

    :type data: pandas DataFrame
    """
    df = data.loc[:, features]
    print('begin filling...')
    for ele in features:
        if show_info == True:
            print(ele)
        p = df.loc[:, ele]
        p[p.isna()] = p[p.notna()].mean()
        df[ele] = p
    print('finished! merge result...')
    for ele in features:
        if show_info == True:
            print(ele)
        data[ele] = df.loc[:, ele]
    print('successflly done!')
    return data


# fill data set missing value with mean (integer mean)
def fill_with_int_mean(data, features, show_info=False):
    """

    :type data: pandas DataFrame
    """
    df = data.loc[:, features]
    print('begin filling...')
    for ele in features:
        if show_info == True:
            print(ele)
        p = df.loc[:, ele]
        p[p.isna()] = int(p[p.notna()].mean())
        df[ele] = p
    print('finished! merge result...')
    for ele in features:
        if show_info == True:
            print(ele)
        data[ele] = df.loc[:, ele]
    print('successflly done!')
    return data


# fill datas et missing value with most frequent value
def fill_with_mode(data, features, show_info=False):
    """

    :type data: pandas DataFrame
    """
    df = data.loc[:, features]
    print('begin filling...')
    for ele in features:
        if show_info == True:
            print(ele)
        p = df.loc[:, ele]
        p[p.isna()] = p[p.notna()].mode()[0]
        df[ele] = p
    print('finished! merge result...')
    for ele in features:
        if show_info == True:
            print(ele)
        data[ele] = df.loc[:, ele]
    print('successflly done!')
    return data


# fill data set missing value with median
def fill_with_median(data):
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    if len(data.shape) == 1:
        data_new = imp.fit_transform(data.values.reshape(-1, 1))
    else:
        data_new = imp.fit_transform(data.values)
    data.values = data_new
    return data


# Using K-nearest neighborhood on non-missing value to test perfomance between features
def KNN_performance(data, train_feature_list, target_feature, k_list=[1, 3, 5, 10, 20, 50, 100, 200, 500],
                    model='class'):
    predictor = data[train_feature_list]
    target = data[target_feature]
    index_na = target[target.isna()].index.values
    index_notna = target[target.notna()].index.values

    X_train = predictor.loc[index_notna, :].values
    y_train = target.loc[index_notna].values
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    knn_result = list()
    for k in k_list:

        if model == 'class':
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            result = accuracy_score(y_test, y_pred)
            print(k, result)
            knn_result.append(result)

        if model == 'reg':
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_train = scaler.fit_transform(X_train)
            y_train = scaler.fit_transform(y_train.ravel())
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            result = mean_squared_error(y_test, y_pred)
            print(k, result)
            print(k, result)
            knn_result.append(result)

    if model == 'class':
        best_result = max(knn_result)
    if model == 'reg':
        best_result = min(knn_result)
    for index, element in knn_result:
        if element == best_result:
            best_k = k_list[index]
    print('best knn result:')
    print(best_k, ' ', best_result)
    return best_k, best_result


# Using kNN predict missing value
def fill_with_KNN_(data, best_k, train_feature_list, target_feature, model='class'):
    predictor = data[train_feature_list]
    target = data[target_feature]
    index_na = target[target.isna()].index.values
    index_notna = target[target.notna()].index.values

    X_train = predictor.loc[index_notna, :].values
    y_train = target.loc[index_notna].values
    X_test = predictor.loc[index_na, :].values

    print('begin train ......')
    if model == 'class':
        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
    if model == 'reg':
        knn = KNeighborsRegressor(n_neighbors=best_k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
    print('successful preidct')

    p1 = target.loc[index_na]
    p2 = pd.Series(y_pred, index=p1.index.values)
    p3 = target[target.notna()]
    p2 = p2.append(p3)
    p2 = p2.sort_index(axis=0)
    data[target_feature] = p2
    print('successfully deal with missing value with data.')

    return data
