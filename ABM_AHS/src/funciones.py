import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
import sklearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb 
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    classification_report, confusion_matrix, roc_curve, auc,
    silhouette_score, make_scorer, precision_recall_curve, roc_auc_score
)

import time


def get_variable_types(dataset = None):
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    
    list_category_variables = []
    list_continuous_variables = []
    
    for i in dataset.columns:
        if (dataset[i].dtype != np.float64) and (dataset[i].dtype != np.int64):
            list_category_variables.append(i)
        else:
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos == 2:
                list_category_variables.append(i)
            else:
                list_continuous_variables.append(i)
          
       
           
    return list_category_variables, list_continuous_variables


def plot_feature(df, col_name, isContinuous, target):
    
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 3), dpi=90)

    count_null = df[col_name].isnull().sum()

    if isContinuous:
        sns.histplot(df.loc[df[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(x=df[col_name], order=sorted(df[col_name].unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name + ' Numero de nulos: ' + str(count_null))
    plt.xticks(rotation=90)

    if isContinuous:
        sns.boxplot(x=col_name, y=target, data=df, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(col_name + ' by ' + target)
    else:
        data = df.groupby(col_name)[target].value_counts(normalize=True).to_frame('proportion').reset_index()
        data.columns = [col_name, target, 'proportion']
        sns.barplot(x=col_name, y='proportion', hue=target, data=data, saturation=1, ax=ax2)
        ax2.set_ylabel(target + ' fraction')
        ax2.set_title(target)
        plt.xticks(rotation=90)
    ax2.set_xlabel(col_name)

    plt.tight_layout()


def get_deviation_of_mean_perc(df, list_continuous_variables, target, multiplier):
    """
    Devuelve el porcentaje de valores que exceden del intervalo de confianza
    :type series:
    :param multiplier:
    :return:
    """
    df_final = pd.DataFrame()
    
    for i in list_continuous_variables:
        
        series_mean = df[i].mean()
        series_std = df[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = df[i].size
        
        perc_goods = df[i][(df[i] >= left) & (df[i] <= right)].size/size_s
        perc_excess = df[i][(df[i] < left) | (df[i] > right)].size/size_s
        
        if perc_excess > 0:    
            df_concat_percent = pd.DataFrame(df[target][(df[i] < left) | (df[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            df_concat_percent.columns = ["no_fraud", "fraud"]
            df_concat_percent = df_concat_percent.drop(df_concat_percent.index[0])
            df_concat_percent['variable'] = i
            df_concat_percent['sum_outlier_values'] = df[i][(df[i] < left) | (df[i] > right)].size
            df_concat_percent['porcentaje_sum_outlier_values'] = perc_excess
            df_final = pd.concat([df_final, df_concat_percent], axis = 0).reset_index(drop = True)
            
    if df_final.empty:
        print('No existen variables con valores nulos')
        
    return df_final

def get_percent_null_values_target(df, list_continuous_variables, target):

    df_final = pd.DataFrame()
    for i in list_continuous_variables:
        if i in ["prev_address_months_count", "current_address_months_count", "bank_months_count",
            "session_length_in_minutes", "device_distinct_emails_8w", "intended_balcon_amount"]:
            df_concat_percent = pd.DataFrame(df[target][df[i].isnull()]\
                                            .value_counts(normalize = True).reset_index()).T
            df_concat_percent.columns = ["no_fraud", "fraud"]
            df_concat_percent = df_concat_percent.drop(df_concat_percent.index[0])
            df_concat_percent['variable'] = i
            df_concat_percent['sum_null_values'] = df[i].isnull().sum()
            df_concat_percent['porcentaje_sum_null_values'] = df[i].isnull().sum()/df.shape[0]
            df_final = pd.concat([df_final, df_concat_percent], axis=0).reset_index(drop=True).sort_values(by='porcentaje_sum_null_values', ascending=False)

    if df_final["sum_null_values"].sum() == 0:
        return print('No existen variables con valores nulos')
        
    return df_final

def get_corr_matrix(dataset=None, method='pearson', size_figure=[10, 8]):
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print('\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="darkgrid")

    # Compute the correlation matrix
    corr = dataset.corr(method=method)

    # Set self-correlation to NaN to exclude diagonal elements
    np.fill_diagonal(corr.values, np.nan)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5, cmap='viridis')

    plt.show()

    return 0


def evaluate_model(model, X_train, y_train, X_val, y_val, threshold=0.5):
    start_time = time.time()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred > threshold).astype(int)
    
    accuracy = accuracy_score(y_val, y_pred_binary)
    precision = precision_score(y_val, y_pred_binary)
    recall = recall_score(y_val, y_pred_binary)
    f1 = f1_score(y_val, y_pred_binary)
    f2 = fbeta_score(y_val, y_pred_binary, beta=2)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return accuracy, precision, recall, f1, f2, execution_time  
