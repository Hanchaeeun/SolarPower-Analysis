# Change to Day, Time Data Format
def Format_PV(df, date, id_cols):
  data = df.copy()
  data = data.melt(id_vars= id_cols , var_name='시간', value_name='발전량')
  data['시간'] = data['시간'].astype(int)
  data.sort_values(['지역', date,'시간'], inplace=True)
  data.reset_index(drop=True, inplace=True)
  return data

# Separation of data by region
def Split_area(data, col, area_list):
  area_name = ['Mp', 'Gn', 'Jj' ] 
  for idx, area in enumerate(area_list): 
    name = area_name[idx]
    area_df = data[data[col] == area]
    area_df.reset_index(drop=True, inplace=True)
    globals()[f'{name}_df'] = area_df
  
  Mp = Mp_df[col].unique()
  Gn = Gn_df[col].unique()
  Jj = Jj_df[col].unique()

  print(f'전체 : {data.shape}\n{Mp} : {Mp_df.shape}\n{Gn} : {Gn_df.shape}\n{Jj} : {Jj_df.shape}')
  return Mp_df, Gn_df, Jj_df 