import numpy as np
import pandas as pd
from math import sqrt
import tensorflow as tf
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler 
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from keras.layers import Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
# model
import lightgbm as lgb
from keras.layers import LSTM, GRU
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential


# train, test_set 생성함수
def Make_DataSet(df, RNN=None, dev=None): 
   # 마지막 년도 test로 사용
  test_year = df['Date'].str[:4].unique().max()
  last_year = df[df['Date'] >= f'{test_year}-01-01'].index[0]
  train = df[df.index < last_year]  
  test = df[df.index >= last_year]

  cols = [x for x in df.columns if x not in ['Date','Area','Month','Time']]
  feature_cols = [x for x in cols if x not in ['Photovoltaics']]

  if dev is True:
    feature_train = [f'dev_{x}' for x in feature_cols]
    mean_train, dev_df = deviation_col(train, cols)
    x_train = dev_df[feature_train]
    y_train = dev_df[['dev_Photovoltaics']]

    mn_df = pd.merge(test[['Month','Time','Photovoltaics']], mean_train[['Month','Time','Photovoltaics']], on=['Month','Time'], how='left')
    mn_df.rename(columns = {'Photovoltaics_y':'mean_Photovoltaics'}, inplace=True)
    testmn = mn_df[['mean_Photovoltaics']]
    testmn = testmn.fillna(0)

    # test-set
    mean_test, dev_df = deviation_col(test, cols)
    x_test = dev_df[feature_train]

  else:
    x_train = train[feature_cols]
    y_train = train[['Photovoltaics']]
    x_test = test[feature_cols]
    testmn = 0
  # test target
  y_test = test[['Photovoltaics']]

  print('Train-set: ', ', '.join(x_train.columns), y_train.columns,'\nTest-set : ', ', '.join(x_test.columns), y_test.columns) 
  
  if RNN == 'RNN':
    testmn, x_train, y_train, x_test, y_test = Make_RNN(x_train, x_test, y_train, y_test, feature_cols, testmn, dev)
    
  print(f'train: {x_train.shape}, {y_train.shape} test: {x_test.shape}, {y_test.shape}\n')
  return testmn, x_train, y_train, x_test, y_test


def Fold_RNN(df, train_idx, valid_idx, dev=None): 

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

  validmn, x_train, y_train, x_valid, y_valid = Make_RNN(x_train, x_valid, y_train, y_valid, feature_cols, validmn, dev)
  print(f'train: {x_train.shape}, {y_train.shape} valid: {x_valid.shape}, {y_valid.shape}\n')
  return validmn, x_train, y_train, x_valid, y_valid

def Search_RNN(df, params_dic, model_name, dev=None):

  testmn, x_train, y_train, x_test, y_test = Make_DataSet(df, 'RNN', dev)
  train_shape = x_train.shape[2]
  
  if model_name == 'LSTM':
    estimator = KerasRegressor(build_fn=create_LSTM(shape=train_shape), verbose=0)
  elif model_name == 'GRU':
    estimator = KerasRegressor(build_fn=create_GRU(shape=train_shape), verbose=0)

  param_grid = {
    'epochs': [25, 30, 35, 40],
    'dense_nparams': [10, 16, 32, 64, 128],
    'batch_size':[12, 16, 32, 64],
    'optimizer': ['Adam', 'Adamax'],
    'dropout': [0.5, 0.4, 0.3, 0.2, 0.1],
    'activation' : ['relu', 'tanh']
  }
  search_cv = RandomizedSearchCV(estimator, param_grid, n_iter=10, cv=2, verbose=1, n_jobs = -1)
  search_cv.fit(x_train, y_train)
  if dev is True:
    params_dic[f'dev_{model_name}'] = search_cv.best_params_
  else:
    params_dic[f'ori_{model_name}'] = search_cv.best_params_
  return params_dic

def Model_RNN(df, model_name, params, score, shap_dic, dev=None):

  cols = [x for x in df.columns if x not in ['Date','Area','Month','Time']]
  feature_ori = [x for x in cols if x not in ['Photovoltaics']]
  feature_dev = [f'dev_{x}' for x in cols if x not in ['']]

  testmn, train_x ,train_y ,test_x ,test_y  = Make_DataSet(df, 'RNN', dev)
  train_shape = train_x.shape[2]
  #tf 초기화
  init_op = tf.compat.v1.global_variables_initializer()
  sess = tf.compat.v1.Session()
  sess.run(init_op)
  if dev is True:
    dev_name = 'dev'
  else:
    dev_name = 'ori'

  # tunning model
  if model_name == 'LSTM' :
    final_model = create_LSTM(train_shape,
                              params[f'{dev_name}_LSTM']['optimizer'], 
                              params[f'{dev_name}_LSTM']['dropout'],
                              params[f'{dev_name}_LSTM']['dense_nparams'],
                              params[f'{dev_name}_LSTM']['activation'])
  elif model_name == 'GRU':
    final_model = create_GRU(train_shape,
                             params[f'{dev_name}_GRU']['optimizer'], 
                             params[f'{dev_name}_GRU']['dropout'],
                             params[f'{dev_name}_GRU']['dense_nparams'],
                             params[f'{dev_name}_GRU']['activation'])

  #5-fold
  tscv = TimeSeriesSplit(n_splits = 5)

  for train_index, valid_index, in tscv.split(train_x ,train_y):
    X_train, X_valid = train_x[train_index, :, :], train_x[valid_index, :, :]
    Y_train, Y_valid = train_y[train_index, :], train_y[valid_index, :]

    print(f'\ntrain_set : {X_train.shape}, valid_set : {X_valid.shape}')
    final_model.fit(X_train, Y_train, epochs=params[f'{dev_name}_{model_name}']['epochs'], batch_size=params[f'{dev_name}_{model_name}']['batch_size'], validation_split=0.2, verbose=0)

  # test
  y_pred = final_model.predict(test_x)
  if dev is True:
    y_pred = y_pred + testmn 
    # score 
    evs = explained_variance_score(y_pred = y_pred, y_true = test_y)
    mape = mean_absolute_percentage_error(y_pred=y_pred, y_true=test_y)
    score[f'EVS_dev_{model_name}'] = evs
    score[f'MAPE_dev_{model_name}'] = mape
    shap_dic[f'dev_{model_name}'] = {'shap_model' : final_model, 'train_X' : train_x, "model" :  model_name, 'cols' : feature_dev}
  else:
    evs = explained_variance_score(y_pred=y_pred, y_true=test_y)
    mape = mean_absolute_percentage_error(y_pred=y_pred, y_true=test_y)
    score[f'EVS_{model_name}'] = evs
    score[f'MAPE_{model_name}'] = mape
    shap_dic[f'ori_{model_name}'] = {'shap_model' : final_model, 'train_X' : train_x, "model" :  model_name, 'cols' : feature_ori}

  return y_pred


### inner Function 

def create_LSTM(shape, optimizer="adam", dropout=0.1, dense_nparams=256, activation='relu'):
  model = Sequential()
  model.add(LSTM(dense_nparams, activation=activation, input_shape=(20, shape)))
  model.add(Dense(dense_nparams, activation=activation)) 
  model.add(Dropout(dropout), )
  model.add(Dense(1))
  model.compile(loss='mean_absolute_percentage_error', optimizer=optimizer)
  return model

def create_GRU(shape, optimizer="adam", dropout=0.1, dense_nparams=256, activation='relu'):
  model = Sequential()
  model.add(GRU(dense_nparams, activation=activation, input_shape=(20, shape)))
  model.add(Dense(dense_nparams, activation=activation)) 
  model.add(Dropout(dropout), )
  model.add(Dense(1))
  model.compile(loss='mean_absolute_percentage_error', optimizer=optimizer)
  return model


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

def Make_RNN(x_train, x_test, y_train, y_test, feature_cols, testmn, dev=None):
  sc = MinMaxScaler() 
  train_sc = sc.fit_transform(x_train)
  test_sc = sc.transform(x_test)

  train_sc_df = pd.DataFrame(train_sc, columns=feature_cols, index=x_train.index)
  test_sc_df = pd.DataFrame(test_sc, columns=feature_cols, index=x_test.index)

  train_X, train_Y = make_sequence(train_sc_df, y_train, 20)
  test_X, test_Y = make_sequence(test_sc_df, y_test, 20)

  if dev is True:
    t, testmn = make_sequence(test_sc_df, testmn, 20)

  return testmn, train_X ,train_Y ,test_X ,test_Y 

def make_sequence(X_Data, Y_Data, sequence):
  feature_list = []
  target_list = []
  for i in range(len(X_Data)-sequence):
    X = np.array(X_Data.iloc[i:i+sequence])
    Y = np.array(Y_Data.iloc[i+sequence])
    feature_list.append(X)
    target_list.append(Y)

  return np.array(feature_list), np.array(target_list)

