import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
import scipy.stats as stats
import json
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from functools import reduce
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import boxcox
from sklearn.model_selection import cross_val_score
import category_encoders as ce
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone

from sklearn.model_selection import cross_validate




class Extraction(TransformerMixin):
    def __init__(self,column):
        self.column = column
        
    def fit(self,df,y=None):
        print('-- Extraction of ',self.column)
        return self
    
    def transform(self,df):
        df_copy = df.copy()
        for index,row in enumerate(df_copy[self.column]):
            if not pd.isna(row):
                d = json.loads(row.replace("'", "\""))
                for key,val in d.items():
                    df_copy.loc[index,key] = val
            
        return df_copy



class StringReplacer(TransformerMixin):
    def __init__(self,columns, source, target):
        self.columns = columns
        self.source = source
        self.target = target
        
    def fit(self,df,y=None):
        print('-- Replacing: **',self.source,'** to: ',self.target,' , for: ',self.columns)
        return self
    
    def transform(self,df):
        df_copy = df.copy()
        for col in self.columns:
            sample = df_copy[col].str.contains(self.source, na=False, regex=False, case=False)
            df_copy.loc[sample, col] = self.target
        return df_copy

class ObjectToNumeric(TransformerMixin):
    def __init__(self,columns):
        self.columns = columns
        
    def fit(self,df,y=None):
        print('-- Transform object to numeric for: ',self.columns)
        return self
    
    def transform(self,df):
        df_copy = df.copy()
        for col in self.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        return df_copy

class ReplaceOutliersWithPercentile(TransformerMixin):
    def __init__(self, col_names):
        self.col_names = col_names

    def fit(self, df, y=None, **fit_params):
        
        print('-- Replace outliers with percentile for: ',self.col_names)
        
        self.min_extreme = {}
        self.max_extreme = {}
        self.min_replace = {}
        self.max_replace = {}
        for col in self.col_names:
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.min_replace[col] = df[col].quantile(0.05)
            self.max_replace[col] = df[col].quantile(0.95)
            self.min_extreme[col] = (Q1 - 1.5 * IQR)
            self.max_extreme[col] = (Q3 + 1.5 * IQR)                   
        return self
        

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        
        for col in self.col_names:
            extremes_min = df_copy[df_copy[col] < self.min_extreme[col]].index
            extremes_max = df_copy[df_copy[col] > self.max_extreme[col]].index
            df_copy.loc[extremes_min, col] = self.min_replace[col]
            df_copy.loc[extremes_max, col] = self.max_replace[col]
        return df_copy

class ReplaceOutliers(TransformerMixin):
    def __init__(self, col_names, min_extreme, max_extreme, min_replace=None, max_replace=None):
        self.col_names = col_names
        self.min_extreme = min_extreme
        self.max_extreme = max_extreme
        self.min_replace = {} if min_replace == None else min_replace
        self.max_replace = {} if max_replace == None else max_replace

    def fit(self, df, y=None, **fit_params):
        
        print('-- Replace outliers with defined extremes for: ',self.col_names)
        
        # ak nie su zadene hodnoty na nahradenie, nahradza sa 5 a 95 percentilnou hodnotou
        if(not self.min_replace and not self.max_replace):
            for col in self.col_names:

                self.min_replace[col] = df[col].quantile(0.05)
                self.max_replace[col] = df[col].quantile(0.95)
        return self
        

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        
        for col in self.col_names:
            extremes_min = df_copy[df_copy[col] < self.min_extreme[col]].index
            extremes_max = df_copy[df_copy[col] > self.max_extreme[col]].index
            df_copy.loc[extremes_min, col] = self.min_replace[col]
            df_copy.loc[extremes_max, col] = self.max_replace[col]
        return df_copy

class DropOutliers(TransformerMixin):
    def __init__(self, col_names):
        self.col_names = col_names

    def fit(self, df, y=None, **fit_params):
        print('-- Drop outliers for: ',self.col_names)
        self.min_extreme = {}
        self.max_extreme = {}
        for col in self.col_names:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.min_extreme[col] = (Q1 - 1.5 * IQR)
            self.max_extreme[col] = (Q3 + 1.5 * IQR)                   
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for col in self.col_names:
            extremes_min = df_copy[df_copy[col] < self.min_extreme[col]].index
            extremes_max = df_copy[df_copy[col] > self.max_extreme[col]].index
            df_copy.drop(extremes_min, inplace=True)
            df_copy.drop(extremes_max, inplace=True)
        return df_copy

class ZScoreNormalization(TransformerMixin):
    def __init__(self, col_names, new_name=""):
        self.col_names = col_names

    def fit(self, df, y=None, **fit_params):
        self.mean = {}
        self.std = {}
        for col in self.col_names:
            self.mean[col] = df[col].mean()
            self.std[col] = df[col].std()
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for col in self.col_names:
            transformed = (df[col] - self.mean[col])/ self.std[col]
            df_copy[col] = transformed
        return df_copy

class LogNormalization(TransformerMixin):
    def __init__(self, col_names, new_name=""):
        self.col_names = col_names

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for col in self.col_names:
            transformed = df[col].apply(lambda x: math.log(x))
            df_copy[col] = transformed
        return df_copy

class BoxCoxNormalization(TransformerMixin):
    def __init__(self, col_names, new_name=""):
        self.col_names = col_names

    def fit(self, df, y=None, **fit_params):
        self.boxcox_attr = {}
        for col in self.col_names:
            _, self.boxcox_attr[col] = boxcox(df[col])
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for col in self.col_names:
            transformed = boxcox(df_copy[col], lmbda=self.boxcox_attr[col])
            df_copy[col] = transformed
        return df_copy


class Scale(TransformerMixin):
    def __init__(self, col_names, scaler):
        self.col_names = col_names
        self.scaler = scaler

    def fit(self, df, y=None, **fit_params):
        self.scalers = {}
        for col in self.col_names:
            self.scalers[col] = self.scaler.fit(df[col])
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for col in self.col_names:
            df_copy[col] = self.scalers[col].transform(df_copy[col])
        return df_copy


class ReplaceNans(TransformerMixin):
    def __init__(self, col_names, func_type = 'median'):
        self.col_names = col_names
        self.func_type = func_type

    def fit(self, df, y=None, **fit_params):
        self.na_replace = {}
        for col in self.col_names:
            if(self.func_type == 'median'):
                self.na_replace[col] = df[col].median()
            else:
                self.na_replace[col] = df[col].mean()
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for col in self.col_names:
            df_copy[col] = df_copy[col].fillna(self.na_replace[col])
        return df_copy

class ReplaceNanWithModel(TransformerMixin):
    def __init__(self, col_names, predict_model, predict_columns):
        self.col_names = col_names
        self.models = {}
        for col in col_names:
            self.models[col] = clone(predict_model)
        self.predict_columns = predict_columns

    def fit(self, df, y=None, **fit_params):
        self.replacers = {}
        for col in self.col_names:
            newdf = df[self.predict_columns]
            newdf = newdf.dropna()  

            self.replacers[col] = ReplaceNans(col_names = newdf[newdf.columns.difference([col])].columns)
            self.replacers[col] = self.replacers[col].fit(newdf)
            
            scores = cross_val_score(self.models[col], 
                                                      newdf[newdf.columns.difference([col])],
                                                     df.loc[newdf.index,col], 
                                                      scoring='neg_mean_squared_error', cv=5)

            # pre kontrolu vypiseme metriku pre urcenie pribliznej ocakavanej uspesnosti modelu
            print("pocet hodnot: " + str(len(newdf)) + "/" + str(len(df)) + ", "
                  + col + "(neg_mean_squared_error):",scores.mean()) 

            self.models[col] = self.models[col].fit(newdf[newdf.columns.difference([col])],newdf[col])
        return self

    def transform(self, df, **transform_params):
        print("transform ", self.col_names)
        df_copy = df.copy()
        
        for col in self.col_names:
            # ziskanie vsetkych riadkov ktore chceme predikovat
            newdf = df[df[col].isna()]
            print("pocet predikovanych pre ", col, ": ", len(newdf))

            if (len(newdf) == 0):
                continue
                
            newdf = newdf[self.predict_columns]
            
            # doplnenie medianu do vsetkych riadkov kde su nejake nan hodnoty okrem atributu ktory predikujeme
            newdf = self.replacers[col].transform(newdf)

            prediction = self.models[col].predict(newdf[newdf.columns.difference([col])])
            df_copy.loc[newdf.index,col] = prediction
        return df_copy


class ReplaceMostFrequent(TransformerMixin):
    def __init__(self, col_names):
        self.col_names = col_names

    def fit(self, df, y=None, **fit_params):
        self.na_replace = {}
        for col in self.col_names:
            self.na_replace[col] = df[col].mode()
          
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for col in self.col_names:
            df_copy[col] = df_copy[col].fillna(self.na_replace[col][0])
        return df_copy


class EncodeCategories(TransformerMixin):
    def __init__(self, encoder):
        self.encoder = encoder

    def fit(self, df, y=None, **fit_params):
        self.encoder.fit(df)
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        df_copy = self.encoder.tranform(df)
        return df_copy


class ReplaceCategoryNanWithModel(TransformerMixin):
        
    def __init__(self, col_names, predict_columns, predict_model,encoder):
        self.col_names = col_names
        self.models = {}
        self.encoders = {}
        for col in col_names:
            self.models[col] = clone(predict_model)
            self.encoders[col] = clone(encoder)
            
        self.predict_columns = predict_columns
        

    def fit(self, df, y=None, **fit_params):
        
        self.replacers = {}
        for col in self.col_names:
            # vynatie len potrebnych stlpcov na ktorych budeme trenovat 
            newdf = df[self.predict_columns]
            newdf = newdf.dropna()
            
            # vynatie trenovacej sady a labels
            newdf_x = newdf[newdf.columns.difference([col])]
            newdf_y = newdf[col]
            
            # natrenovanie classy ktora nahradzuje chybajuce kategoricke atrinuty pri transformaccii
            self.replacers[col] = ReplaceMostFrequent(col_names = newdf_x.columns)
            self.replacers[col] = self.replacers[col].fit(newdf)
            
            # encodovanie kategorickych atrinutov na ciselne 
            self.encoders[col].fit(newdf_x)
            newdf_x = self.encoders[col].transform(newdf_x)
            
            
            scores = cross_val_score(self.models[col], 
                                                      newdf_x,
                                                      newdf_y, 
                                                      scoring='accuracy', cv=5)

            # pre kontrolu vypiseme metriku pre urcenie pribliznej ocakavanej uspesnosti modelu
            print("pocet hodnot: " + str(len(newdf)) + "/" + str(len(df)) + ", "
                  + col + "(accuracy score):",scores.mean()) 

            self.models[col] = self.models[col].fit(newdf_x,newdf_y)
        return self

    def transform(self, df, **transform_params):
    
        print("transform ", self.col_names)
        df_copy = df.copy()
        
        for col in self.col_names:
            # ziskanie vsetkych riadkov ktore chceme predikovat
            newdf = df[df[col].isna()]
            print("pocet predikovanych pre ", col, ": ", len(newdf))

            if (len(newdf) == 0):
                continue
                
            # ziskanie len potrebnych atributov
            newdf = newdf[self.predict_columns]
            
            newdf_x = newdf[newdf.columns.difference([col])]
            
            # doplnenie do vsetkych riadkov kde su nejake nan hodnoty okrem atributu ktory predikujeme
            newdf_x = self.replacers[col].transform(newdf_x)
            
            newdf_x = self.encoders[col].transform(newdf_x)

            prediction = self.models[col].predict(newdf_x)
            df_copy.loc[newdf_x.index,col] = prediction
        return df_copy
    

class Selector(TransformerMixin):
    def __init__(self, columns):
        self.columns_to_select = columns

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        df_copy = df[self.columns_to_select]
        return df_copy


# funkcia reduce ktora z hodnot v stlpci vrati hodnotu ak tam nejaka je ,inak NaN
def func(vstup):
    return reduce(lambda x,y: x if not pd.isna(x) else y, vstup)

# funkcia odstrani deduplikaty
def deduplicate(df,columns = [], func=sum):
    df_copy = df.copy()
    # najdeme vsetky deduplikaty, a pomocou groupby a agregacnej funkcie ziskame originalny zaznam
    deduplicated = df_copy[df_copy.duplicated(subset=columns, keep=False)].groupby(columns).agg(func).reset_index()
    # ziskame vsetky zaznamy  ktore nemaju duplikaty
    df_copy.drop_duplicates(subset=columns, keep=False, inplace=True)
    #spojime tieto dva dataframy(opravene duplikaty, originaly)
    return pd.concat([df_copy,deduplicated], sort = True).reset_index(drop=True)



def merge_and_deduplicate(df1, df2, columns=[],deduplic=[], func=sum):
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    
    if 0 in deduplic:
        df1_copy = deduplicate(df1_copy, columns, func)
    if 1 in deduplic:
        df2_copy = deduplicate(df2_copy, columns, func)
    
    return pd.merge(df1_copy, df2_copy, on=columns)
 