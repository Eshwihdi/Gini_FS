import pandas as pd
from pandas import DataFrame
import csv
  
  
def Gini(df):
  #df = df.select_dtypes(include=[np.number])
  df_col = df[[ i for i in list(df.columns) if i not in ['Weight','PG','PB']]]
  cols = df_col.columns
  record_collinear = None
  record_collinear = pd.DataFrame(columns = ['Field', 'Gini'])

  for n in cols:
    df1 = df[[n,'Weight','PG','PB']] 
    df1 = df1.sort_values([n])
    df1['TotG'] = df1['Weight'] * df1['PG']
    df1['TotB'] = df1['Weight'] * df1['PB']
    SumTotG = df1['TotG'].sum()
    SumTotB = df1['TotB'].sum()

    df1['CumulativeG'] = df1['TotG'].cumsum(axis = 0)
    df1['CumulativeB'] = df1['TotB'].cumsum(axis = 0)

    LastCumulativeG = df1['CumulativeG'].iloc[-1]
    LastCumulativeB = df1['CumulativeB'].iloc[-1]

    df1['CumulativeG%']=df1['CumulativeG']/SumTotG
    df1['CumulativeB%']=df1['CumulativeB']/SumTotB
    df1['G(i)'] = pd.rolling_sum(df1['CumulativeG%'],2)
    df1['B(i)'] = df1['CumulativeB%'].diff()
    df1['Gini'] = df1['G(i)'] * df1['B(i)']
    df1 = df1.fillna(0)
    Gini_I = (1 - df1['Gini'].sum()).round(4)
    temp_df = pd.DataFrame.from_dict({'Field': [n],'Gini': [Gini_I]})
    record_collinear = record_collinear.append(temp_df, ignore_index = True)

  record_collinear = record_collinear
  record_collinear = record_collinear.sort_values(by="Gini", ascending=False)
  record_collinear.to_csv('Data/Gini.csv')
  print (record_collinear)
