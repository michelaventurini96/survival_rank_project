import os
import time
from tqdm import tqdm
import torch
import torchtuples as tt
import numpy as np
import easydict
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from RANKSURV.model import CombinedLossSurvModel
from RANKSURV.metrics import concordance_index_censored
from RANKSURV.surv_data import SurvData, collate_fn
from torch.utils.data import DataLoader
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
import xgboost as xgb


### HELPER FUNCTIOND
def get_labels_rsf(t, e):
    
    y = np.zeros(len(t), dtype = {'names': ('e', 't'),
                                            'formats': ('bool', 'i4')})

    y['e'] = e > 0
    y['t'] = t

    return y

def surv_xgb(y):
    
    y_xgb = [t[0] if t[1] else -t[0] for t in y.values]
    
    return y_xgb

def get_x_xgb(X, y):
    
    y_xgb = [x[0] if x[1] else -x[0] for x in y.values]

    xgb_data = xgb.DMatrix(X, label=y_xgb)

    return xgb_data

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

get_target = lambda df: (df['duration'].values, df['event'].values)


### FITTING FUNCTIONS
def train_rank_surv_custom(x_train, time_train, event_train, x_test, time_test, event_test, cidxtype, measure, alpha, lr, nn):
        # SET PARAMETERS
    args = easydict.EasyDict({
        "batch_size": 1024,
        "cuda": False,
        "lr": lr,
        "seed": 1111,
        "reduce_rate": 0.95,
        "epochs": 200,
        "clip": 5.0,
        "log_interval":10,
    })

    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.FloatTensor
    if torch.cuda.is_available() and args.cuda:
        dtype = torch.cuda.FloatTensor

    np.random.seed(1234)
    _ = torch.manual_seed(123)
    
    # initialize model
    in_features = x_train.shape[1]
    num_nodes = nn
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, output_bias=output_bias)


    model  = CombinedLossSurvModel(net, sigma=0.01, Cindex_type=cidxtype,
                        event_train= event_train, alpha=alpha,
                        time_train = time_train, measure=measure, dtype=dtype).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=args.reduce_rate)


    # load data
    train_data = SurvData(x_train, time_train, event_train)
    train_load = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    train_loader = iter(cycle(train_load))

    test_data = SurvData(x_test, time_test, event_test)
    test_load = DataLoader(
        dataset=test_data,
        batch_size=x_test.shape[0],
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = iter(cycle(test_load))

    train_iter = len(train_data) // args.batch_size +1
    test_iter = 1

    # Start model training and evaulation
    # cindex_epoch = []
    # mae_epoch = []
    for epoch in range(args.epochs):
        start_time = time.time()

        for i_iter in range(train_iter):
            model.train()
            x, y, event = next(train_loader)
            x = torch.from_numpy(x).to(device).type(dtype)
            loss, _,_,_= model(x, y, event)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            # if i_iter % args.log_interval == 0 and i_iter > 0:
            #     elapsed = time.time() - start_time
            #     cur_loss = loss.item()
            #     print('epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
            #           'current loss {:5.4f} '.format(
            #               epoch, i_iter, train_iter, elapsed * 1000 / args.log_interval,cur_loss))
            #     start_time = time.time()

        event_indicator = np.array([])
        event_time = np.array([])
        estimate = np.array([])
        for i_test in range(test_iter):
            model.eval()
            x_test, y_test, event_test = next(test_loader)
            x_test = torch.from_numpy(x_test).to(device).type(dtype)
            _, estimate_y, _, _ = model(x_test, y_test, event_test)
            event_indicator = np.hstack((event_indicator, event_test))
            event_time = np.hstack((event_time,y_test))
            estimate = np.hstack((estimate,estimate_y.cpu().detach().numpy()))

        test_cindex = concordance_index_censored(event_indicator.astype(bool), event_time, -1*estimate)
        cindex,_,_,_,_ = test_cindex
        # mae = sum(abs(estimate-event_time)*event_indicator)/sum(event_indicator)

    return cindex

def train_rank_surv_mod(x_train, time_train, event_train, cidxtype, measure, alpha, lr, nn):
        # SET PARAMETERS
    args = easydict.EasyDict({
        "batch_size": 1024,
        "cuda": False,
        "lr": lr,
        "seed": 1111,
        "reduce_rate": 0.95,
        "epochs": 200,
        "clip": 5.0,
        "log_interval":10,
    })

    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.FloatTensor
    if torch.cuda.is_available() and args.cuda:
        dtype = torch.cuda.FloatTensor

    np.random.seed(1234)
    _ = torch.manual_seed(123)
    
    # initialize model
    in_features = x_train.shape[1]
    num_nodes = nn
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, output_bias=output_bias)


    model  = CombinedLossSurvModel(net, sigma=0.01, Cindex_type=cidxtype,
                        event_train= event_train, alpha=alpha,
                        time_train = time_train, measure=measure, dtype=dtype).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=args.reduce_rate)


    # load data
    train_data = SurvData(x_train, time_train, event_train)
    train_load = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    train_loader = iter(cycle(train_load))


    train_iter = len(train_data) // args.batch_size +1

    # Start model training and evaulation
    # cindex_epoch = []
    # mae_epoch = []
    for epoch in range(args.epochs):
        start_time = time.time()

        for i_iter in range(train_iter):
            model.train()
            x, y, event = next(train_loader)
            x = torch.from_numpy(x).to(device).type(dtype)
            loss, _,_,_= model(x, y, event)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            # if i_iter % args.log_interval == 0 and i_iter > 0:
            #     elapsed = time.time() - start_time
            #     cur_loss = loss.item()
            #     print('epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
            #           'current loss {:5.4f} '.format(
            #               epoch, i_iter, train_iter, elapsed * 1000 / args.log_interval,cur_loss))
            #     start_time = time.time()

    return model

def train_deepsurv(x_train, time_train, event_train, x_val, time_val, event_val, lr, nn):
    
    np.random.seed(1234)
    _ = torch.manual_seed(123)
    
    in_features = x_train.shape[1]
    num_nodes = nn
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                dropout, output_bias=output_bias)
    
    batch_size = 1024
    epochs = 200
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(lr)
    
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = False
    
    df_train = pd.DataFrame(columns=['duration', 'event'])
    df_train.loc[:, 'duration'] = time_train
    df_train.loc[:, 'event'] = event_train
    
    df_val = pd.DataFrame(columns=['duration', 'event'])
    df_val.loc[:, 'duration'] = time_val
    df_val.loc[:, 'event'] = event_val
    
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    val = x_val, y_val
    
    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
    
    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_val)
    
    ev = EvalSurv(surv, time_val, event_val, censor_surv='km')
    cidx = ev.concordance_td()
    
    return cidx

def train_deepsurv_mod(x_train, time_train, event_train, lr, nn):
    
    np.random.seed(1234)
    _ = torch.manual_seed(123)
    
    in_features = x_train.shape[1]
    num_nodes = nn
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                dropout, output_bias=output_bias)
    
    batch_size = 1024
    epochs = 200
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(lr)
    
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = False
    
    df_train = pd.DataFrame(columns=['duration', 'event'])
    df_train.loc[:, 'duration'] = time_train
    df_train.loc[:, 'event'] = event_train
    
    y_train = get_target(df_train)
    
    train = x_train, y_train
    
    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=train, val_batch_size=batch_size)
    
    _ = model.compute_baseline_hazards()
    return model
    

### TUNING FUNCTIONS
def tune_params_surv(x_train, time_train, event_train, x_val, time_val, event_val):
    
    y_train = get_labels_rsf(time_train, event_train)
    
    print('COX')
    s_cox           = np.zeros((9, ))     
    l1_ratio        = [.3, .5, .9]
    alpha_min_ratio = [.00001, .001, .1]
    
    i = 0
    for l1 in l1_ratio:
        for a in alpha_min_ratio:
            m = CoxnetSurvivalAnalysis(l1_ratio=l1, alpha_min_ratio=a).fit(x_train, y_train)
            e = m.predict(x_val)
            s_cox[i] = concordance_index_censored(event_indicator=event_val.astype(bool), event_time=time_val, estimate=e)[0]
            
            print('l1: ' + str(l1) + ', a: '+str(a)+' score: '+str(s_cox[i]))
            i+=1
        
    print('RSF')
    s_rsf             = np.zeros((9, ))
    n_estimators      = [100, 200, 300]
    max_depth = [6, 9, 12]
    
    i = 0
    for n_e in n_estimators:
        for mss in max_depth:
            m = RandomSurvivalForest(n_estimators=n_e, max_depth=mss).fit(x_train, y_train)
            e = m.predict(x_val)
            s_rsf[i] = concordance_index_censored(event_indicator=event_val.astype(bool), event_time=time_val, estimate=e)[0]
            
            print('n_est: ' + str(n_e) + ', max_d: '+str(mss)+' score: '+str(s_rsf[i]))
            i+=1
    
    print('XGB')
    s_xgb       = np.zeros((9, ))
    eta         = [0.003, 0.03, 0.3]
    max_depth   = [6, 9, 12]
    
    i=0
    for et in eta:
        for md in max_depth:  
            param       = {'objective': 'survival:cox', 'eta': et, 'max_depth': md}
            m = xgb.train(param, get_x_xgb(x_train, pd.DataFrame({'t': time_train, 'e': event_train})))
            e = m.predict(xgb.DMatrix(x_val))
            e[e == np.inf] = e[e != np.inf].max()
            e[e == -np.inf] = e[e != -np.inf].min()
            e[e == np.nan] = e.mean()
            s_xgb[i] = concordance_index_censored(event_indicator=event_val.astype(bool), event_time=time_val, estimate=e)[0]
            
            print('eta: ' + str(et) + ', max_d: '+str(md)+' score: '+str(s_xgb[i]))
            i+=1
            
    print('DEEPSURV')
    s_ds     = np.zeros((9, ))     
    lrs       = [.05]
    num_nodes = [[16], [32], [64], [16,16], [32,32], [64,64], [16,16,16], [32,32,32], [64,64,64]]
    
    i = 0
    for lr in lrs:
        for nn in num_nodes:
            s_ds[i] = train_deepsurv(x_train.astype('float32'), time_train.values.astype('float32'), event_train.values.astype(int), \
                x_val.astype('float32'), time_val.values.astype('float32'), event_val.values.astype(int), lr, nn)
            
            print('lr: '+str(lr)+ ', nn: '+str(nn)+', score: '+str(s_ds[i]))
            i+=1
                    
    print(np.argmax(s_ds))
    
    print('RANKSURV')
    s_rs     = np.zeros((36, ))     
    cidxtypes = ['Harrell']
    measures  = ['mse']
    alpha     = .5
    
    i = 0
    for cidxtype in cidxtypes:
        for measure in measures:
            for lr in lrs:
                for nn in num_nodes:
                    s_rs[i] = train_rank_surv_custom(x_train, time_train.values, event_train.values, x_val, time_val.values, event_val.values, cidxtype, measure, alpha, lr, nn)
                    
                    print('CIDXT: ' + str(cidxtype) + ', measure: '+str(measure)+ ', lr: '+str(lr)+ ', nn: '+str(nn)+', score: '+str(s_rs[i]))
                    i+=1
                    
    print(np.argmax(s_rs))
    
    return


### MAIN FUNCTION
def main_tune_baselines():
    
    # datasets
    # files = [ 'addicts.csv', 'echocardiogram.csv', 'employee_attrition.csv', 'flchain.csv', 'gabs.csv', 'GBSG2.csv', \
    #     'lung.csv', 'metabric.csv', 'NSBDC.csv', 'nwtco.csv', 'primary_biliary_cirrhosis.csv', 'rotterdam.csv', \
    #         'support.csv', 'Telco-CLT.csv', 'Telco-CLV.csv', 'veteran.csv']

    # files = ['nwtco.csv', 'primary_biliary_cirrhosis.csv', 'rotterdam.csv', \
    #         'support.csv', 'Telco-CLT.csv', 'Telco-CLV.csv', 'veteran.csv']
    
    files = ['rotterdam.csv', 'support.csv', 'Telco-CLT.csv', 'Telco-CLV.csv', 'veteran.csv']
    
    for file in tqdm(files):
        print('FILE: ', file)
        # GET DATA
        path = os.getcwd()+'/data/'
        file_name = path+file
        data = pd.read_csv(file_name)
        
        
        # PREPROCESS DATA
        X = data.iloc[:, :-2]
        
        if file == 'veteran.csv':
            X = pd.get_dummies(X, columns=['Celltype'])
            
        if file == 'primary_biliary_cirrhosis.csv':
            X = pd.get_dummies(X, columns=['sex'])
            
        if file == 'GBSG2.csv':
            X = pd.get_dummies(X, columns=['horTh', 'tgrade', 'menostat'])
            
        if file == 'rotterdam.csv':
            X = pd.get_dummies(X, columns=['size'])
        
        
        X= X.fillna(X.median())
        
        X_normalize = preprocessing.scale(X)

        time_all = data.iloc[:, -2].fillna(0).round(0).astype(int)
        event_all = data.iloc[:, -1]
        
        x_train1, x_test, event_train1, event_test = train_test_split(X_normalize, event_all,
                                                                stratify=event_all, 
                                                                test_size=0.2,
                                                                random_state=2436)

        time_train1, time_test = time_all.loc[event_train1.index], time_all.loc[event_test.index]
        
        x_train, x_val, event_train, event_val = train_test_split(x_train1, event_train1,
                                                                stratify=event_train1, 
                                                                test_size=0.2,
                                                                random_state=2436)

        time_train, time_val = time_train1.loc[event_train.index], time_train1.loc[event_val.index]
        
        tune_params_surv(x_train, time_train, event_train, x_val, time_val, event_val)

def main_test_perf_baselines():
    
    # datasets
    files = [ 'addicts.csv', 'employee_attrition.csv', 'flchain.csv', 'gabs.csv', 'GBSG2.csv', \
        'lung.csv', 'metabric.csv', 'nwtco.csv', 'primary_biliary_cirrhosis.csv', 'rotterdam.csv', \
            'support.csv', 'Telco-CLT.csv', 'Telco-CLV.csv', 'veteran.csv']
    
    param_tuned = pd.read_excel('parameters_baselines.xlsx')
    
    f = 0
    for file in tqdm(files):
        print('FILE: ', file)
        # GET DATA
        path = os.getcwd()+'/data/'
        # file  = 'nwtco.csv'
        file_name = path+file
        data = pd.read_csv(file_name)
        
        
        # PREPROCESS DATA
        X = data.iloc[:, :-2]
        
        if file == 'veteran.csv':
            X = pd.get_dummies(X, columns=['Celltype'])
            
        if file == 'primary_biliary_cirrhosis.csv':
            X = pd.get_dummies(X, columns=['sex'])
            
        if file == 'GBSG2.csv':
            X = pd.get_dummies(X, columns=['horTh', 'tgrade', 'menostat'])
            
        if file == 'rotterdam.csv':
            X = pd.get_dummies(X, columns=['size'])
        
        X= X.fillna(X.median())
        
        X_normalize = preprocessing.scale(X)

        time_all = data.iloc[:, -2].fillna(0).round(0).astype(int)
        event_all = data.iloc[:, -1]
        
        x_train, x_test, event_train, event_test = train_test_split(X_normalize, event_all,
                                                                stratify=event_all, 
                                                                test_size=0.2,
                                                                random_state=2436)

        time_train, time_test = time_all.loc[event_train.index], time_all.loc[event_test.index]
        y_train = get_labels_rsf(time_train, event_train)
        
        n_bootstraps = 100
        rng_seed = 2436 
        cidx_cox = np.zeros((n_bootstraps, ))
        cidx_rsf = np.zeros((n_bootstraps, ))
        cidx_xgb = np.zeros((n_bootstraps, ))
        cidx_dps = np.zeros((n_bootstraps, ))
        cidx_rns = np.zeros((n_bootstraps, ))
        
        l1_ratio        = param_tuned.iloc[0, f+1]
        alpha_min_ratio = param_tuned.iloc[1, f+1]
        mcox = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=alpha_min_ratio).fit(x_train, y_train)
        
        n_estimators = param_tuned.iloc[2, f+1].astype(int)
        max_depth    = param_tuned.iloc[3, f+1].astype(int)
        mrsf = RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth).fit(x_train, y_train)
        
        eta = param_tuned.iloc[4, f+1]
        max_depth = param_tuned.iloc[5, f+1].astype(int)
        param       = {'objective': 'survival:cox', 'eta': eta, 'max_depth': max_depth}
        mxgb = xgb.train(param, get_x_xgb(x_train, pd.DataFrame({'t': time_train, 'e': event_train})))
        
        lr = 0.01
        l = param_tuned.iloc[6, f+1].astype(int)
        num_nodes = param_tuned.iloc[7, f+1].astype(int)
        num_nodes = [num_nodes for _ in range(l)]
        
        mdps = train_deepsurv_mod(x_train.astype('float32'), time_train.values.astype('float32'), event_train.values.astype(int), lr, num_nodes)

        l = param_tuned.iloc[8, f+1].astype(int)
        num_nodes = param_tuned.iloc[9, f+1].astype(int)
        num_nodes = [num_nodes for _ in range(l)]
        cidxtype  = 'Harrell'
        measure   = 'mse'
        alpha =.5
        mrns = train_rank_surv_mod(x_train, time_train.values, event_train.values, cidxtype, measure, alpha, lr, num_nodes)
        
        time_test = time_test.values
        event_test = event_test.values
            
        rng = np.random.RandomState(rng_seed)
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(time_test), len(time_test))
            if len(np.unique(event_test[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            
            x_test_boot     = x_test[indices]
            time_test_boot  = time_test[indices]
            event_test_boot = event_test[indices]
        
            y_train = get_labels_rsf(time_train, event_train)
        
            e1 = mcox.predict(x_test_boot)
            cidx_cox[i] = concordance_index_censored(event_indicator=event_test_boot.astype(bool), event_time=time_test_boot, estimate=e1)[0]
                
            e2 = mrsf.predict(x_test_boot)
            cidx_rsf[i] = concordance_index_censored(event_indicator=event_test_boot.astype(bool), event_time=time_test_boot, estimate=e2)[0]
            
            e3 = mxgb.predict(xgb.DMatrix(x_test_boot))
            e3[e3 == np.inf] = e3[e3 != np.inf].max()
            e3[e3 == -np.inf] = e3[e3 != -np.inf].min()
            e3[e3 == np.nan] = e3.mean()
            cidx_xgb[i] = concordance_index_censored(event_indicator=event_test_boot.astype(bool), event_time=time_test_boot, estimate=e3)[0]
                    
            e4 = mdps.predict(x_test_boot.astype('float32')).reshape(e3.shape[0], )
            
            cidx_dps[i] = concordance_index_censored(event_indicator=event_test_boot.astype(bool), event_time=time_test_boot, estimate=e4)[0]

            device = torch.device("cpu")
            dtype = torch.FloatTensor
            
            test_data = SurvData(x_test_boot, time_test_boot, event_test_boot)
            test_load = DataLoader(
                dataset=test_data,
                batch_size=x_test_boot.shape[0],
                shuffle=True,
                collate_fn=collate_fn
            )

            test_loader = iter(cycle(test_load))
            test_iter = 1       
            event_indicator = np.array([])
            event_time = np.array([])
            estimate = np.array([])
            for i_test in range(test_iter):
                mrns.eval()
                x_test_boot, y_test, event_test = next(test_loader)
                x_test_boot = torch.from_numpy(x_test_boot).to(device).type(dtype)
                _, estimate_y, _, _ = mrns(x_test_boot, y_test, event_test)
                event_indicator = np.hstack((event_indicator, event_test))
                event_time = np.hstack((event_time,y_test))
                e5 = np.hstack((estimate,estimate_y.cpu().detach().numpy()))

            cidx_rns[i] = concordance_index_censored(event_indicator.astype(bool), event_time, -1*e5)[0]
        
        # confidence intervals
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100  
        p1 = (alpha+((1.0-alpha)/2.0)) * 100
        
        # cox
        lower = max(0.0, np.percentile(cidx_cox, p))
        upper = min(1.0, np.percentile(cidx_cox, p1))
        print('COX: ' + 'Mean = ' + str(cidx_cox.sum()/n_bootstraps) + ', CI = [' + str(lower*100) + ', ' + str(upper*100) + ']')
        
        # rsf
        lower = max(0.0, np.percentile(cidx_rsf, p))
        upper = min(1.0, np.percentile(cidx_rsf, p1))
        print('RSF: ' + 'Mean = ' + str(cidx_rsf.sum()/n_bootstraps) + ', CI = [' + str(lower*100) + ', ' + str(upper*100) + ']')
        
        # xgb
        lower = max(0.0, np.percentile(cidx_xgb, p))
        upper = min(1.0, np.percentile(cidx_xgb, p1))
        print('XGB: ' + 'Mean = ' + str(cidx_xgb.sum()/n_bootstraps) + ', CI = [' + str(lower*100) + ', ' + str(upper*100) + ']')
        
        # dps
        lower = max(0.0, np.percentile(cidx_dps, p))
        upper = min(1.0, np.percentile(cidx_dps, p1))
        print('DPS: ' + 'Mean = ' + str(cidx_dps.sum()/n_bootstraps) + ', CI = [' + str(lower*100) + ', ' + str(upper*100) + ']')
        
        # rns
        lower = max(0.0, np.percentile(cidx_rns, p))
        upper = min(1.0, np.percentile(cidx_rns, p1))
        print('RNS: ' + 'Mean = ' + str(cidx_rns.sum()/n_bootstraps) + ', CI = [' + str(lower*100) + ', ' + str(upper*100) + ']')
        
        f += 1
        
def main_test_perf_baselines1():
    
    # datasets
    files = [ 'addicts.csv', 'employee_attrition.csv', 'flchain.csv', 'gabs.csv', 'GBSG2.csv', \
        'lung.csv', 'metabric.csv', 'nwtco.csv', 'primary_biliary_cirrhosis.csv', 'rotterdam.csv', \
            'support.csv', 'Telco-CLT.csv', 'Telco-CLV.csv', 'veteran.csv']
    
    param_tuned = pd.read_excel('parameters_baselines.xlsx')
    
    f = 0
    for file in tqdm(files):
        print('FILE: ', file)
        # GET DATA
        path = os.getcwd()+'/data/'
        # file  = 'nwtco.csv'
        file_name = path+file
        data = pd.read_csv(file_name)
        
        
        # PREPROCESS DATA
        X = data.iloc[:, :-2]
        
        if file == 'veteran.csv':
            X = pd.get_dummies(X, columns=['Celltype'])
            
        if file == 'primary_biliary_cirrhosis.csv':
            X = pd.get_dummies(X, columns=['sex'])
            
        if file == 'GBSG2.csv':
            X = pd.get_dummies(X, columns=['horTh', 'tgrade', 'menostat'])
            
        if file == 'rotterdam.csv':
            X = pd.get_dummies(X, columns=['size'])
        
        X= X.fillna(X.median())
        
        X_normalize = preprocessing.scale(X)

        time_all = data.iloc[:, -2].fillna(0).round(0).astype(int)
        event_all = data.iloc[:, -1]
        
        x_train, x_test, event_train, event_test = train_test_split(X_normalize, event_all,
                                                                stratify=event_all, 
                                                                test_size=0.2,
                                                                random_state=2436)

        time_train, time_test = time_all.loc[event_train.index], time_all.loc[event_test.index]
        y_train = get_labels_rsf(time_train, event_train)
        
        cidx_cox = 0
        cidx_rsf = 0
        cidx_xgb = 0
        cidx_dps = 0
        cidx_rns = 0
        
        l1_ratio        = param_tuned.iloc[0, f+1]
        alpha_min_ratio = param_tuned.iloc[1, f+1]
        mcox = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=alpha_min_ratio).fit(x_train, y_train)
        
        n_estimators = param_tuned.iloc[2, f+1].astype(int)
        max_depth    = param_tuned.iloc[3, f+1].astype(int)
        mrsf = RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth).fit(x_train, y_train)
        
        eta = param_tuned.iloc[4, f+1]
        max_depth = param_tuned.iloc[5, f+1].astype(int)
        param       = {'objective': 'survival:cox', 'eta': eta, 'max_depth': max_depth}
        mxgb = xgb.train(param, get_x_xgb(x_train, pd.DataFrame({'t': time_train, 'e': event_train})))
        
        lr = 0.01
        l = param_tuned.iloc[6, f+1].astype(int)
        num_nodes = param_tuned.iloc[7, f+1].astype(int)
        num_nodes = [num_nodes for _ in range(l)]
        
        mdps = train_deepsurv_mod(x_train.astype('float32'), time_train.values.astype('float32'), event_train.values.astype(int), lr, num_nodes)

        l = param_tuned.iloc[8, f+1].astype(int)
        num_nodes = param_tuned.iloc[9, f+1].astype(int)
        num_nodes = [num_nodes for _ in range(l)]
        cidxtype  = 'Harrell'
        measure   = 'mse'
        alpha =.5
        mrns = train_rank_surv_mod(x_train, time_train.values, event_train.values, cidxtype, measure, alpha, lr, num_nodes)
        
        time_test = time_test.values
        event_test = event_test.values
            
        
        y_train = get_labels_rsf(time_train, event_train)
    
        e1 = mcox.predict(x_test)
        cidx_cox = concordance_index_censored(event_indicator=event_test.astype(bool), event_time=time_test, estimate=e1)[0]
            
        e2 = mrsf.predict(x_test)
        cidx_rsf = concordance_index_censored(event_indicator=event_test.astype(bool), event_time=time_test, estimate=e2)[0]
        
        e3 = mxgb.predict(xgb.DMatrix(x_test))
        e3[e3 == np.inf] = e3[e3 != np.inf].max()
        e3[e3 == -np.inf] = e3[e3 != -np.inf].min()
        e3[e3 == np.nan] = e3.mean()
        cidx_xgb = concordance_index_censored(event_indicator=event_test.astype(bool), event_time=time_test, estimate=e3)[0]
                
        e4 = mdps.predict(x_test.astype('float32')).reshape(e3.shape[0], )
        
        cidx_dps = concordance_index_censored(event_indicator=event_test.astype(bool), event_time=time_test, estimate=e4)[0]

        device = torch.device("cpu")
        dtype = torch.FloatTensor
        
        test_data = SurvData(x_test, time_test, event_test)
        test_load = DataLoader(
            dataset=test_data,
            batch_size=x_test.shape[0],
            shuffle=True,
            collate_fn=collate_fn
        )

        test_loader = iter(cycle(test_load))
        test_iter = 1       
        event_indicator = np.array([])
        event_time = np.array([])
        estimate = np.array([])
        for i_test in range(test_iter):
            mrns.eval()
            x_test, y_test, event_test = next(test_loader)
            x_test = torch.from_numpy(x_test).to(device).type(dtype)
            _, estimate_y, _, _ = mrns(x_test, y_test, event_test)
            event_indicator = np.hstack((event_indicator, event_test))
            event_time = np.hstack((event_time,y_test))
            e5 = np.hstack((estimate,estimate_y.cpu().detach().numpy()))

        cidx_rns = concordance_index_censored(event_indicator.astype(bool), event_time, -1*e5)[0]
        
        # cox
        print('COX: ' + str(cidx_cox))
        
        # rsf
        print('RSF: ' + str(cidx_rsf))
        
        # xgb
        print('XGB: ' + str(cidx_xgb))
        
        # dps
        print('DPS: ' + str(cidx_dps))
        
        # rns
        print('RNS: ' + str(cidx_rns))
    
        f += 1
    
if __name__ == "__main__":
    
    main_test_perf_baselines1()