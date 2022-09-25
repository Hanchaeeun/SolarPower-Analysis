# import package
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import chain, combinations
import matplotlib.pyplot as plt

def Correlation(area_df, cols, name):
  fig, ax = plt.subplots(figsize=(12,8), dpi=100)
  plt.title(f'{name} correlation', fontsize=15)
  
  data = area_df.copy()
  data = data[cols]

  corr_df = data.corr()
  mask = np.zeros_like(corr_df, dtype=bool)
  mask[np.triu_indices_from(mask)] = True

  sns.heatmap(corr_df, 
            cmap = 'RdBu_r', 
            annot = True,   
            mask = mask,
            linewidths = .5,
            cbar_kws = {"shrink": .8}, 
            vmin = -1,
            vmax = 1
             )  
  plt.xticks(rotation=15)
  plt.show()
  return corr_df


def CorrBar(corr_df, cols, area):
  Sublist = chain.from_iterable(combinations(cols,n) for n in range(2,3))
  data = pd.DataFrame(columns=["cols",'corr'])
  for c in Sublist:
    corr_num = corr_df.loc[c]
    add_df = pd.DataFrame([[c, corr_num]] , columns=["cols",'corr'])
    data = pd.concat([data, add_df])
  data.sort_values('corr', ascending=False, inplace=True)
  data.reset_index(inplace=True, drop=True)

  pos_df = data[data['corr'] >= 0.3] 
  neg_df = data[data['corr'] < -0.3]

  # plot
  fig, [ax1, ax2] = plt.subplots(figsize=(25,15), nrows=2, ncols=1)
  bar1 = ax1.bar(range(0, len(pos_df)), pos_df['corr'], color=sns.color_palette('Reds_r', len(pos_df)))
  bar2 = ax2.bar(range(0, len(neg_df)), -neg_df['corr'], color=sns.color_palette('Blues', len(neg_df)))

  plt.suptitle(f'negative and positive correlation in {area}', fontsize=30, y=0.95)
  ax1.set_xticks(range(0, len(pos_df)))
  ax2.set_xticks(range(0, len(neg_df)))
  ax1.set_xticklabels([i for i in pos_df['cols'].str[0]], fontsize=15, rotation=-25)
  ax2.set_xticklabels([i for i in neg_df['cols'].str[0]], fontsize=15, rotation=-25)

  #labeld
  po_label = [i for i in pos_df['cols'].str[1]]
  ne_label = [i for i in neg_df['cols'].str[1]]
  for bar in [bar1,bar2]:
    if bar == bar1:
      label = po_label
      ax = ax1
    else:
      label = ne_label
      ax = ax2
    idx=0  
    for b in bar:
      x = b.get_x()
      y = b.get_height()
      ax.text(x , y, label[idx], va = 'bottom', fontsize=15)
      idx+=1
  plt.show()
  return data


def SeasonSplit(area_df): 
  hot_season = pd.DataFrame()
  cold_season = pd.DataFrame()
  # high-temperature month
  for i in range(5,11):
    idx = area_df[area_df['Month'] == i].index
    data = area_df.loc[idx]
    hot_season = pd.concat([hot_season,data])
  # low-temperature month
  for i in [1,2,3,4,11,12]: 
    idx = area_df[area_df['Month'] == i].index
    data = area_df.loc[idx]
    cold_season = pd.concat([cold_season,data])
  hot_season.reset_index(drop=True, inplace=True)
  cold_season.reset_index(drop=True, inplace=True)
  return hot_season, cold_season