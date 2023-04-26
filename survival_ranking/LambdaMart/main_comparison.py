from lambdamart_surv import LambdaMART
from lambdamart_cens import LambdaMARTC
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import time
from tqdm import tqdm

def get_labels_rsf(t, e):
    
    y = np.zeros(len(t), dtype = {'names': ('e', 't'),
                                            'formats': ('bool', 'i4')})

    y['e'] = e > 0
    y['t'] = t

    return y


def get_data(time, event, x):
    
    data = []
    for i in range(len(time)):
        new_arr = []
        new_arr.append(int(time[i]))
        new_arr.append(int(event[i]))
        arr = x[i, :]
        for el in arr:
            new_arr.append(float(el))
        data.append(new_arr)
    return np.array(data)

files = ['veteran.csv', 'addicts.csv', 'lung.csv', 'primary_biliary_cirrhosis.csv']

# files = [ 'addicts.csv', 'employee_attrition.csv', 'flchain.csv', 'gabs.csv', 'GBSG2.csv', \
#         'lung.csv', 'metabric.csv', 'nwtco.csv', 'primary_biliary_cirrhosis.csv', 'rotterdam.csv', \
#             'support.csv', 'Telco-CLT.csv', 'Telco-CLV.csv', 'veteran.csv']

# files = ['GBSG2.csv']


for file in files:
    path = os.getcwd()+'/../../data/'
    file_name = path+file
    data = pd.read_csv(file_name)

    # PREPROCESS DATA
    X = data.iloc[:, :-2]
    
    if file == 'veteran.csv':
        X = pd.get_dummies(X, columns=['Celltype'])
        nt = 300
        lnr = .3
    
    if file == 'lung.csv':
        nt = 300
        lnr = .003
    
    if file == 'addicts.csv':
        nt = 100
        lnr = .03
    
    if file == 'primary_biliary_cirrhosis.csv':
        nt = 200
        lnr = .003
        
    if file == 'primary_biliary_cirrhosis.csv':
        X = pd.get_dummies(X, columns=['sex'])
        
    if file == 'GBSG2.csv':
        X = pd.get_dummies(X, columns=['horTh', 'tgrade', 'menostat'])
        nt = 300
        lnr = .003
    if file == 'rotterdam.csv':
        X = pd.get_dummies(X, columns=['size'])
    
    X = X.fillna(X.median())
    
    X_normalize = preprocessing.scale(X)
    time_all = data.iloc[:, -2].values
    event_all = data.iloc[:, -1].values
    
    time_all = data.iloc[:, -2].fillna(0).round(0).astype(int)
    event_all = data.iloc[:, -1]
        
    x_train, x_test, event_train, event_test = train_test_split(X_normalize, event_all,
                                                            stratify=event_all, 
                                                            test_size=0.2,
                                                            random_state=2436)

    time_train, time_test = time_all.loc[event_train.index], time_all.loc[event_test.index]
    
    training_data = get_data(time_train.values, event_train.values, x_train)
    test_data 	  = get_data(time_test.values, event_test.values, x_test)
    
    start_time = time.time()
    model = LambdaMART(training_data, number_of_trees=nt, learning_rate=lnr, tree_type='sklearn')
    model.fit()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    perf_our = concordance_index_censored(event_test.astype(bool), time_test, model.predict(test_data))[0]
    perf_our_tr = concordance_index_censored(event_train.astype(bool), time_train, model.predict(training_data))[0]

    print('Dataset: ', file)
    print('Data shape: ', X.shape)
    print('Our: ' + 'Test ' + str(perf_our) + ' - Train ' + str(perf_our_tr))