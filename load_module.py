# import package
import os
import pathlib
import re
import numpy as np
import pandas as pd

# Loading PV Data in Jinju and Gangneung
def Load_PV(path):
  file_list = os.listdir(path)

  result_df = pd.DataFrame()
  for i in file_list:
    df = pd.read_csv(path+i, encoding='CP949')
    df['년도'] = pd.to_datetime(df['년월일']).dt.year
    result_df = pd.concat([result_df, df], axis=0)
  result_df.drop(columns=['호기', '구분', '총량', '평균', '최대', '최소', '최대(시간별)', '최소(시간별)'], inplace=True)  
  return result_df


# Loading PV Data in Jinju and Gangneung
def Load_PV_Mokpo(path):
  file_list = os.listdir(path)

  result_df = pd.DataFrame()
  for i in file_list:
    df = pd.read_csv(path+i, encoding='CP949')
    year = i[-8:-4]
    df['년도'] = np.int(year)
    df.dropna(how='all', axis=1, inplace=True)

    cols = [x for x in df.columns if x not in [' 계 ','년도','월','일']]
    for i in cols:
      rname = re.sub(r'[^0-9]',"",i)
      df.rename(columns = {i:np.str(np.int(rname))}, inplace=True)
    result_df = pd.concat([result_df, df], axis=0)
  result_df.drop(columns=' 계 ', inplace=True)
  return result_df


# Loading ASOS Data
def Load_ASOS(path):
  file_list = os.listdir(path)
  df = pd.DataFrame()

  for i in file_list:
    print(i)
    ASOSdata = pd.read_csv(path+i, encoding='CP949')
    df = pd.concat([df, ASOSdata])
  df = df.sort_values('일시').reset_index(drop=True)

  df['시간'] = pd.to_datetime(df['일시']).dt.hour
  df['년도'] = pd.to_datetime(df['일시']).dt.year

  col_list = df.columns
  col_rename = []
  for i in col_list:
    name = i.strip().split("(")[0]
    col_rename.append(name)
  df.columns = col_rename 
  return df


# Loading Particulate Data
def Load_PM(path, name, area):
  file_list = os.listdir(path)
  PMdata = pd.DataFrame()
  for i in file_list:
    p = pathlib.Path(path+i)
    Extension = p.suffix
    if Extension == '.xlsx':
      data = pd.read_excel(p)
    else:
      try:
        data = pd.read_csv(p, encoding = 'CP949')
      except:
        data = pd.read_csv(p, encoding = 'utf8')
    data = data[data[name].isin(area)]
    PMdata = pd.concat([PMdata, data]).reset_index(drop=True)

  PMdata['시간'] = PMdata['측정일시'].astype(str).str[-2:]
  PMdata['시간'] = PMdata['시간'].astype(int)
  PMdata['일자'] = PMdata['측정일시'].astype(str).str[:-2]
  PMdata['년도'] = PMdata['측정일시'].astype(str).str[:4]

  PMdata.sort_values(['일자'], inplace=True)
  return PMdata