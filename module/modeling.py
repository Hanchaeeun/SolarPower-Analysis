import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler 
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from keras.layers import Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from rnn_modeling import Make_DataSet
# model
import lightgbm as lgb
from keras.layers import LSTM, GRU
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential


def FoldSet(df, train_idx, valid_idx, dev=None): 

  cols = [x for x in df.columns if x not in ['Date','Area','Month','Time']]
  feature_cols = [x for x in cols if x not in ['Photovoltaics']]
  
  train = df.loc[train_idx]  
  valid = df.loc[valid_idx]

  if dev is True:
    feature_train = [f'dev_{x}' for x in feature_cols]
    mean_train, dev_df = deviation_col(train, cols)
    x_train = dev_df[feature_train]
    y_train = dev_df[['dev_Photovoltaics']]

    mn_df = pd.merge(valid[['Month','Time','Photovoltaics']], mean_train[['Month','Time','Photovoltaics']], on=['Month','Time'], how='left')
    mn_df.rename(columns = {'Photovoltaics_y':'mean_Photovoltaics'}, inplace=True)
    validmn = mn_df[['mean_Photovoltaics']]
    validmn = validmn.fillna(0)

    # test-set
    mean_valid, dev_df = deviation_col(valid, cols)
    x_valid = dev_df[feature_train]

  else:
    x_train = train[feature_cols]
    y_train = train[['Photovoltaics']]
    x_valid = valid[feature_cols]
    validmn = 0
  # valid target
  y_valid = valid[['Photovoltaics']]

  if dev is True:
    print(f'train: {x_train.shape}, {y_train.shape} valid: {x_valid.shape}, {y_valid.shape}\n')
  return validmn, x_train, y_train, x_valid, y_valid

def GridSearch_ML(df, cols, params_dic, dev=None):
  # model
  MLP = MLPRegressor()
  LGBM = lgb.LGBMRegressor()
             
  param_mlp = {'solver' : ['adam'],
              'activation' : ['identity', 'logistic', 'relu', 'tanh'],
              'max_iter': [3000, 4000],
              'learning_rate' : ['constant', 'invscaling', 'adaptive'],
              'hidden_layer_sizes': [
                  (100,),(200,),(300,),(400,),(500,)]
              }
  param_lgb = {'learning_rate' : [0.01, 0.1],
               'max_depth' : [-1, -5, 1, 5],
               'objective' : ['regression'],
               'metric' : ['mse'],
               'boosting': ['gbdt', 'rf', 'dart', 'goss'],
               'num_leaves': [10, 25, 31, 35],
               }
  # train, test set
  testmn, x_train, y_train, x_test, y_test = Make_DataSet(df, None , dev)

  if dev is True:
    name = 'dev'
  else:
    name = 'ori'

  # hyperparameter tuning
  MLP_grid = GridSearchCV(estimator = MLP, param_grid = param_mlp, scoring ='r2', cv=2, n_jobs = -1)
  MLP_grid.fit(x_train, y_train.values.ravel())
  params_dic[f'{name}_MLP'] = MLP_grid.best_params_

  LGBM_grid = GridSearchCV(estimator = LGBM , param_grid = param_lgb, scoring ='r2', cv=2, n_jobs = -1)
  LGBM_grid.fit(x_train, y_train)
  params_dic[f'{name}_LGBM'] = LGBM_grid.best_params_
  
  return params_dic

def MLTest(df, model, params_dic, score, shap_dic, dev=None):

  model_list = {'LR': LinearRegression()}

  # tuning hyperparameter
  if dev is True:
    if model == 'LGBM':
      parameter_LGBM = params_dic[f'dev_LGBM']
    elif model == 'MLP':
      parameter_MLP = params_dic[f'dev_MLP']
      model_list['MLP'] = MLPRegressor(**parameter_MLP)
    else:
      pass
  else :
    if model == 'LGBM':
      parameter_LGBM = params_dic[f'ori_LGBM']
    elif model == 'MLP':
      parameter_MLP = params_dic[f'ori_MLP']
      model_list['MLP'] = MLPRegressor(**parameter_MLP)
    else:
      pass
  
  # train, test set
  cols = [x for x in df.columns if x not in ['Date','Area','Month','Time']]
  feature_cols = [x for x in cols if x not in ['Photovoltaics']]

  testmn, x_train, y_train, x_test, y_test = Make_DataSet(df, None , dev)

  # 5-fold
  tscv = TimeSeriesSplit(n_splits = 5)
    
  if model == 'LGBM':

    for train_index, valid_index, in tscv.split(x_train, y_train):
      validmn, train_x, train_y, valid_x, valid_y = FoldSet(df, train_index, valid_index, dev)
      lgb_train = lgb.Dataset(data = train_x, label = train_y)
      LGBM_model = lgb.train(params=parameter_LGBM, train_set = lgb_train)
    # test
    y_pred = LGBM_model.predict(x_test) 
    shap_model = LGBM_model
      
  else:
    for train_index, valid_index, in tscv.split(x_train, y_train):
      validmn, train_x, train_y, valid_x, valid_y = FoldSet(df, train_index, valid_index, dev)
      globals()[f'{model}_model'] = model_list[model].fit(train_x, train_y.values.ravel())
    # test 
    y_pred = globals()[f'{model}_model'].predict(x_test)
    shap_model = globals()[f'{model}_model']

  if dev is True:
    y_pred = y_pred + testmn.squeeze() # Predictive Deviation + Average
    score[f'EVS_dev_{model}'] = explained_variance_score(y_true=y_test, y_pred=y_pred)
    score[f'MAPE_dev_{model}'] = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
    shap_dic[f'dev_{model}'] = {'model' : shap_model, 'x_train' : x_train, 'model_name' : model}
        
  else:
    score[f'EVS_{model}'] = explained_variance_score(y_true=y_test, y_pred=y_pred)
    score[f'MAPE_{model}'] = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
    shap_dic[f'ori_{model}'] = {'model' : shap_model, 'x_train' : x_train, 'model_name' : model}
  
  print(shap_model)
  return y_pred 

def RF_importances(df, dev=True):
  testmn, x_train, y_train, x_test, y_test = Make_DataSet(df, 'ML', dev)

  # model fit
  RF_model = RandomForestRegressor()
  RF_model.fit(x_train, y_train.values.ravel())
  # test
  y_pred = RF_model.predict(x_test)

  if dev is True:
    y_pred = y_pred + testmn.squeeze()
  # score
  evs = explained_variance_score(y_true = y_test, y_pred = y_pred)
  mape = mean_absolute_percentage_error(y_true = y_test, y_pred = y_pred)

  importances = RF_model.feature_importances_

  # sort values
  imp = {}
  for idx, i in enumerate(x_train.columns):
    imp[i] = importances[idx]
  sort_importances = dict(sorted(imp.items(), key=lambda x: x[1]))

  print(f'explained varaiance : {evs}\n MAPE : {mape}')
  return sort_importances

### inner Function 

# Create Deviation Variable
def deviation_col(data, cols): 
  df = data.copy()
  cols.extend(['Month','Time']) 
  # Monthly and Hourly Average
  mean_df = pd.DataFrame() 
  for m in df['Month'].unique():
    for h in df['Time'].unique():
      mn = np.array(df[(df['Month'] == m) & (df['Time'] == h)][cols].mean())
      mean_data = pd.DataFrame([mn], columns = cols)
      mean_df = pd.concat([mean_df, mean_data])

  mean_df.dropna(how='any', inplace=True)
  mean_df.reset_index(inplace=True, drop=True)
  mean_df[['Time','Month']] = mean_df[['Time','Month']].astype(int)

  # Deviation Variable
  cols.remove('Month')
  cols.remove('Time')
  dev_df = pd.DataFrame(0, index = df.index, columns = cols) 

  hours = mean_df['Time'].unique()
  months = mean_df['Month'].unique()
  for m, h in [(x,y) for x in months for y in hours]:
    idx = df[(df['Month'] == m) & (df['Time'] == h)].index
    m_idx = mean_df[(mean_df['Month'] == m) & (mean_df['Time'] == h)].index
    dev_df.loc[idx, cols] = df.loc[idx, cols] - mean_df.loc[m_idx, cols].squeeze()

  # rename
  rename_c = [f'dev_{i}' for i in dev_df.columns]
  dev_df.columns = rename_c

  return mean_df, dev_df

