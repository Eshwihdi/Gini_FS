import pandas as pd
from Gini_Coefficient import Gini
from feature_selector import FeatureSelector

#filename = "Data/PCF Data v2.0.csv"
Target = 'PG'

df_pandas = pd.read_csv("Data/PCF Data v3.0.csv")
df_pandas.info()
df = df_pandas

## Split data:
train = df
train_labels = train[Target]
train.head()
train = train.drop(Target,axis=1)
fs = FeatureSelector(data = train, labels = train_labels)

## Missing Values:
fs.identify_missing(missing_threshold=0.6)
missing_features = fs.ops['missing']
missing_features[:100]
fs.plot_missing()
fs.missing_stats.sample(10)


## Single Unique Value:
fs.identify_single_unique()
single_unique = fs.ops['single_unique']
single_unique

fs.plot_unique()
fs.unique_stats.sample(5)

#df_0['SPA_E1_28'].dtypes

## Collinear (highly correlated) Features:
fs.identify_collinear(correlation_threshold=0.975)
correlated_features = fs.ops['collinear']
correlated_features[:5]

fs.plot_collinear()

fs.identify_collinear(correlation_threshold=0.65)
fs.plot_collinear()

#fs.record_collinear.head(100)
fs.record_collinear.head(100).sort_values('corr_value',ascending=False)
fs.record_collinear.head(100).sort_values('corr_feature',ascending=True)

#fs.record_collinear.sort_values('corr_value',ascending=False)
#fs.record_collinear.sort_values('corr_feature',ascending=True)


corr_Table = fs.record_collinear.sort_values('corr_value',ascending=False)

corr_Table.to_csv('Data/Corr_Table.csv')

Gini(df)
