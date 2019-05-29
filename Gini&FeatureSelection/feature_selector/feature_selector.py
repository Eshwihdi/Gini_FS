import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from pandas import DataFrame
import csv
from itertools import chain

class FeatureSelector():
    
    def __init__(self, data, labels=None):        
        self.data = data
        self.labels = labels
        self.base_features = list(data.columns)
        self.one_hot_features = None
        self.record_missing = None
        self.record_single_unique = None
        self.record_collinear = None
        self.missing_stats = None
        self.unique_stats = None
        self.corr_matrix = None
        self.ops = {}        
        self.one_hot_correlated = False
        record_collinear = None
        
    def identify_missing(self, missing_threshold):        
        self.missing_threshold = missing_threshold
        missing_series = self.data.isnull().sum() / self.data.shape[0]
        self.missing_stats = pd.DataFrame(missing_series).rename(columns = {'index': 'feature', 0: 'missing_fraction'})
        self.missing_stats = self.missing_stats.sort_values('missing_fraction', ascending = False)
        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(columns = 
                                                                                                               {'index': 'feature', 
                                                                                                                0: 'missing_fraction'})
        to_drop = list(record_missing['feature'])
        self.record_missing = record_missing
        self.ops['missing'] = to_drop        
        print('%d features with greater than %0.2f missing values.\n' % (len(self.ops['missing']), self.missing_threshold))                                                                                                               
        print('%d features with greater than %0.2f missing values.\n' % (len(self.ops['missing']), self.missing_threshold))
        
    def identify_single_unique(self):
        """Finds features with only a single unique value. NaNs do not count as a unique value. """

        # Calculate the unique counts in each column
        unique_counts = self.data.nunique()
        self.unique_stats = pd.DataFrame(unique_counts).rename(columns = {'index': 'feature', 0: 'nunique'})
        self.unique_stats = self.unique_stats.sort_values('nunique', ascending = True)
        
        # Find the columns with only one unique count
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(columns = {'index': 'feature', 
                                                                                                                0: 'nunique'})

        to_drop = list(record_single_unique['feature'])
    
        self.record_single_unique = record_single_unique
        self.ops['single_unique'] = to_drop
        
        print('%d features with a single unique value.\n' % len(self.ops['single_unique']))    
    def identify_collinear(self, correlation_threshold, one_hot=False):        
        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot
        if one_hot:
            features = pd.get_dummies(self.data)
            self.one_hot_features = [column for column in features.columns if column not in self.base_features]
            self.data_all = pd.concat([features[self.one_hot_features], self.data], axis = 1)            
            corr_matrix = pd.get_dummies(features).corr()
        else:
            corr_matrix = self.data.corr()        
        self.corr_matrix = corr_matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))        
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
        record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])
        for column in to_drop:
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]    
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                             'corr_feature': corr_features,
                                             'corr_value': corr_values})
            record_collinear = record_collinear.append(temp_df, ignore_index = True)
        self.record_collinear = record_collinear
        self.ops['collinear'] = to_drop        
        print('%d features with a correlation magnitude greater than %0.2f.\n' % (len(self.ops['collinear']), self.correlation_threshold))

    def plot_missing(self):
        if self.record_missing is None:
            raise NotImplementedError("Missing values have not been calculated. Run `identify_missing`")        
        self.reset_plot()
        plt.style.use('seaborn-white')
        plt.figure(figsize = (7, 5))
        plt.hist(self.missing_stats['missing_fraction'], bins = np.linspace(0, 1, 11), edgecolor = 'k', color = 'red', linewidth = 1.5)
        plt.xticks(np.linspace(0, 1, 11));
        plt.xlabel('Missing Fraction', size = 14); plt.ylabel('Count of Features', size = 14); 
        plt.title("Fraction of Missing Values Histogram", size = 16);
                                                                                                               
    def plot_unique(self):
        if self.record_single_unique is None:
            raise NotImplementedError('Unique values have not been calculated. Run `identify_single_unique`')        
        self.reset_plot()
        self.unique_stats.plot.hist(edgecolor = 'k', figsize = (7, 5))
        plt.ylabel('Frequency', size = 14); plt.xlabel('Unique Values', size = 14); 
        plt.title('Number of Unique Values Histogram', size = 16);
                                                                                                               
    def plot_collinear(self, plot_all = False):        
        if self.record_collinear is None:
            raise NotImplementedError('Collinear features have not been idenfitied. Run `identify_collinear`.')        
        if plot_all:
        	corr_matrix_plot = self.corr_matrix
        	title = 'All Correlations'        
        else:
	        # Identify the correlations that were above the threshold
	        # columns (x-axis) are features to drop and rows (y_axis) are correlated pairs
	        corr_matrix_plot = self.corr_matrix.loc[list(set(self.record_collinear['corr_feature'])), 
	                                                list(set(self.record_collinear['drop_feature']))]
	        title = "Correlations Above Threshold"       
        f, ax = plt.subplots(figsize=(10, 8))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr_matrix_plot, cmap=cmap, center=0,
                    linewidths=.25, cbar_kws={"shrink": 0.6})
        ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
        ax.set_yticklabels(list(corr_matrix_plot.index), size = int(160 / corr_matrix_plot.shape[0]));
        ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
        ax.set_xticklabels(list(corr_matrix_plot.columns), size = int(160 / corr_matrix_plot.shape[1]));
        plt.title(title, size = 14)
                                                                                                               
    def reset_plot(self):
        plt.rcParams = plt.rcParamsDefault
        
        
    