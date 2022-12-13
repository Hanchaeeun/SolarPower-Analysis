# import package
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime as dt
import matplotlib.pyplot as plt


def Correlation(area_df, cols, ax):
  data = area_df.copy()
  data = data[cols]

  corr_df = data.corr()

  mask = np.zeros_like(corr_df, dtype=bool)
  mask[np.triu_indices_from(mask)] = True
  mask = mask[1:,:-1]
  corr_data = corr_df.iloc[1:,:-1]

  map = sns.heatmap(corr_data, 
            cmap = 'RdBu_r', 
            annot = True,
            mask = mask,
            linewidths = .5,
            annot_kws={'fontsize': 18},
            cbar_kws = {"shrink": 1.0, 'ticks': [-1.00, -0.50, 0.00, 0.50, 1.00]},
            vmin = -1,
            vmax = 1,
            ax = ax)
  ax.set_xticklabels(cols[:-1], rotation=90, fontsize=22)
  ax.set_yticklabels(cols[1:], rotation=0, fontsize=22)
  ax.tick_params(axis='x', pad=2, width=2)
  return corr_df
  

def Daymean_plot(area_df, col, ax, name='Monthly mean PV power'):
  # monthly
  day_list = area_df['Date'].unique()
  data = pd.DataFrame()
  for day in day_list:
    day_mean = area_df[area_df['Date'] == day][col].mean()
    mean_df = pd.DataFrame([[day, day_mean]], columns=['Date',f'{col}_mean'])
    data = pd.concat([data, mean_df]).reset_index(drop=True)

  data['Month'] = data['Date'].str[5:7]
  data['Month'] = data['Month'].astype(int)

  for i in range(1,13):
    date = dt(year = 2022, month = i, day = 1)
    month = date.strftime('%b')
    globals()[month] = data[data['Month'].isin([i])][f'{col}_mean'].to_numpy(int)

  m_list = np.array([Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec], dtype=object)
  # Monthly plot
  ax.set_title(name, loc='left', fontsize=20, pad=15)
  ax.boxplot(m_list, showmeans=True)
  ax.set_xlabel('Month', fontsize=20, color='#696965', fontweight='semibold')
  ax.set_xticks(range(1,13))
  ax.set_ylabel(f'{col}', fontsize=20, color='#696965', fontweight='semibold')
  ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=18)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)


def Hourly_plot(area_df, col, ax, name='Hourly PV power'):
  for i in range(0, 24):
    globals()[f'h{i}'] = area_df[area_df['Time'] == i][f'{col}'].to_numpy(int)  

  t_list = np.array([h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23], dtype=object)
  # Hourly plot
  ax.set_title(f"{name}", loc='left', fontsize=20, pad=15)
  ax.boxplot(t_list, showmeans=True)
  ax.set_xticklabels(range(0,24), fontsize=16)
  ax.set_ylim([area_df[col].min(), area_df[col].max()])
  ax.set_xlabel('Time', fontsize=20, color='#696965', fontweight='semibold')
  ax.set_ylabel(f'{col}', fontsize=20, color='#696965', fontweight='semibold')
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)


def Monthly_plot(area_df, col, ax, name='Monthly plot'):
  for m in area_df['Month'].unique():
    Month_df = area_df[area_df['Month'] == m]
    date = dt(year = 2022, month = m, day = 1)
    month = date.strftime('%b')
    globals()[month] = Month_df[col].to_numpy(int)

  m_list = np.array([Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec], dtype=object)
  # Monthly plot
  ax.set_title(f"{name}", loc='left', fontsize=20, pad=15)
  ax.boxplot(m_list, showmeans=True)
  ax.set_xlabel('Month', fontsize=20, color='#696965', fontweight='semibold')
  ax.set_xticks(range(1,13))
  ax.set_ylabel(f'{col}', fontsize=20, color='#696965', fontweight='semibold')
  ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=18)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)


def Scatter(area_df, target, name):
  col_list = [x for x in area_df.columns if x not in ['Area', 'Date', 'Time', 'Month','Photovoltaics']]
  plt.figure(figsize=(20,14))
  for idx, col in enumerate(col_list):
    ax = plt.subplot(3, 4, idx+1)
    sns.regplot(x = area_df[target], y = area_df[col] , data = area_df, scatter_kws = {'color':'b', 's':6}, line_kws = {'color':'r'})
    ax.set_title(f'{col}', fontsize = 15)
    ax.set_xlabel(f'{target}', fontsize = 15)
    ax.set_ylabel(f'{col}') 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
  plt.suptitle(f'{name} Scatter Diagram', fontsize = 20, position=(0.5,0.92)) 
  plt.show()


def NullPlot(data, title, ax):
  cols = [x for x in data.columns if x not in ['Area', 'Date', 'Time', 'Month']]
  null_list = data[cols].isnull().sum()    
  # Missing value
  pal = sns.color_palette("Blues", len(cols))
  rank = null_list.argsort().argsort()
  bar = ax.bar(range(1,len(cols)+1), null_list.values, color=np.array(pal)[rank])
  # title 설정
  ax.set_title(title, fontsize=20)
  ax.set_xticks(range(1, len(cols)+1))
  ax.set_xticklabels(cols, rotation=25)
  ax.set_xlabel('variables', fontsize=15)
  ax.set_ylabel('Null', fontsize=15)
  ax.set_ylim(0, data.isnull().sum().max()+1500)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  for b in bar:
    x = b.get_x() + b.get_width()/2.0
    y = b.get_height()
    ax.text(x, y, y, ha='center', va='bottom', fontsize=13)


# Total Hourly bar plot
def TimeBar(df, title, ylim): 
  test_df = df.copy()
  dic = {}
  for i in range(1,13):
    time_df = test_df[test_df['Month'] == i] 
    sum_num = time_df['Time'].sum()
    dic[i] = sum_num
  time_list = dic.items()
  x, y = zip(*time_list)

  pal = sns.color_palette("Blues", len(x)) 
  sorted_month = [x[0] for x in sorted(time_list, key = lambda num:num[1])] 
  zero_array = np.zeros(12, dtype=int) 
  for idx, i in enumerate(sorted_month):
    if sorted_month[idx] == i:
      zero_array[sorted_month[idx]-1] = idx
  rank_list = [i for i in zero_array]

  fig, ax = plt.subplots(figsize=(12,5))
  bar = ax.bar(x, y, alpha=0.7, color= np.array(pal)[rank_list]) 
  plt.title(f'Total monthly hours of {title}', fontsize=20, pad=10)
  plt.xticks(x, [f'{m}' for m in x])
  plt.ylabel('Sum Time')
  plt.ylim([0, ylim])
  plt.xlabel('Month')

  bar = ax.bar(x, y, alpha=0.7, color= np.array(pal)[rank_list])
  for i, b in enumerate(bar):
    x_text = b.get_x() + b.get_width()/2.0
    y_text = b.get_height()
    ax.text(x_text, y_text, f'{y_text}h', ha='center', va='bottom', fontsize=10)
  plt.show()


def Plot_Result(pred_df, area_name, pred_cols):
  date = pred_df['Date'].unique()
  mn_df = pd.DataFrame(date, columns=['Date'])
  mn_list = []
  for day in date:
    mn = pred_df[pred_df['Date'] == day][pred_cols].mean()
    mn_list.append(mn)
  data = pd.DataFrame(mn_list)
  mn_df = pd.concat([mn_df, data], axis=1)
  # index month
  month = pd.to_datetime(mn_df['Date']).dt.month
  month_idx = [month[[0]].index] # 첫번째 달
  for m in month.unique():
    last = month[month == m].iloc[-1:].index
    month_idx.append(last)
  # plot 
  fig, ax = plt.subplots(figsize=(50,20))
  for col in pred_cols:
    if col == 'PV':
      colors ='k'
      ax.plot(mn_df['Date'], mn_df[col], label=f'{col}', color=colors, linewidth=2.0)
    else:
      ax.plot(mn_df['Date'], mn_df[col], label=f'{col}', linewidth=1.5)
  plt.suptitle(f'predict PV power in {area_name}', y=0.95, fontsize=25)
  plt.legend(loc='upper right', fontsize=20, bbox_to_anchor=(0.9, 1.05), ncol=6)
  min_num = mn_df.iloc[:, 1:].min().min()
  max_num = mn_df.iloc[:, 1:].max().max()
  ax.set_ylim([min_num - 20 , max_num + 20])
  ax.set_xticks(month_idx)
  ax.set_ylabel('PV', fontsize=30)
  ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=25)
  ax.set_xlabel('Date', fontsize=30)
  ax.set_xlim([0, len(month)])
  plt.show()


def Result_pred(df, pred_LR, pred_MLP, pred_LGBM, train_mean):
  pred_cols = ['LR_Pred', 'MLP_Pred', 'LGBM_Pred' , 'train_mean']
  pred_df = pd.DataFrame([pred_LR, pred_MLP, pred_LGBM]).T
  pred_df = pd.concat([pred_df, train_mean], axis=1)
  pred_df.columns = pred_cols
  cols = ['Area','Date','Month','Time','Photovoltaics']

  test_month = df['Date'].unique()[-1][:4]
  result_df = df[df['Date'] >= f'{test_month}-01-01'][cols].reset_index(drop=True)
  result_df = pd.concat([result_df, pred_df], axis=1)

  return result_df

def Result_RNN(df, y_test, pred_df, testmn, pred_cols):
  df_test = pd.DataFrame(testmn)
  df_PV =  pd.DataFrame(y_test)
  data = pd.concat([df_PV, pred_df, df_test], axis=1)
  data.columns = pred_cols

  cols = ['Area','Date','Month','Time']
  test_month = df['Date'].unique()[-1][:4]
  result_df = df[df['Date'] >= f'{test_month}-01-01'][cols].reset_index(drop=True)
  result_df = result_df[20:]
  result_df.reset_index(inplace=True, drop=True)
  result_df = pd.concat([result_df, data], axis=1)

  return result_df