import pandas as pd
import numpy as np
from scipy import stats, special
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline, TransformerMixin

# numeric columns
numeric_columns = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea']

# categorical columns
# 'Neighborhood'
categorical_columns = ['MSZoning','Street','LotShape','LotConfig','Condition2','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','Heating','CentralAir','Electrical','KitchenQual','FireplaceQu','GarageType','GarageFinish','GarageQual','SaleType','SaleCondition','GarageYrBltImputed','Neighborhood']

# recreated categorical variable(to drop)
custom_categorical_columns = ['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','LowQualFinSF','Neighborhood']

# created categorical variable
onehot_categorical_columns = ['PoolQC','SaleCondition','SaleType','FireplaceQu','KitchenQual','RoofMatl','Condition2','ExterQual','BsmtQual','Heating','HeatingQC','porch','neighbor']

# numeric variable corr<0.20
#n_drop_feature = ['1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','LowQualFinSF','EnclosedPorch','3SsnPorch','PoolArea','MiscVal','OverallCond','BsmtFullBath','BsmtHalfBath','BedroomAbvGr','KitchenAbvGr']

n_drop_feature = ['EnclosedPorch','3SsnPorch','PoolArea','MiscVal']

# categorical variable chi^2 p-value >0.05
c_drop_feature = ['MSSubClass','Alley','LandContour','Utilities','LandSlope','Condition1','BldgType','HouseStyle','RoofStyle','Exterior1st','Exterior2nd','BsmtFinType1','BsmtFinType2','Functional','GarageCond','PavedDrive','Fence','MiscFeature','OpenPorchSF','ScreenPorch','Neighborhood']

# to numerical columns
to_numerical_columms = ['YrSold', 'MoSold']

neighbor = [['MeadowV','IDOTRR','BrDale','BrkSide','Edwards','OldTown','Sawyer','Blueste','SWISU','NPkVill','NAmes','Mitchel'],['SawyerW','NWAmes','Gilbert','Blmngtn','CollgCr','Crawfor','ClearCr','Somerst','Veenker','Timber'],['StoneBr','NridgHt','NoRidge']]

class NullFiller(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        # Hyelee
        ## Alley : convert NaN => NA
        df['Alley'].fillna('NA',inplace=True)
        ## LotFrontage : convert NaN => ???? 
        # <CASE1> Using LotArea 
        #df_train1['LotFrontage'].fillna(df_train1['LotArea'] *0.007206024910841549,inplace=True)
        # <CASE2> Using Neighborhood 
        df_tmp = df[~df.LotFrontage.isnull()]
        df_md_lotfrontage = df_tmp.groupby('Neighborhood').agg('median')[{'LotFrontage'}]
        df_md_lotfrontage = df_md_lotfrontage.rename(columns={"LotFrontage" : "mdLotFrontage"})
        df_tmp = df_tmp.merge(df_md_lotfrontage, how="inner", on='Neighborhood')
        df = pd.concat([df.reset_index(drop=True).set_index('Neighborhood'),df_md_lotfrontage], axis=1, join='inner').reset_index()
        df['LotFrontage'].fillna(df['mdLotFrontage'],inplace=True)
        df = df.drop(columns=['mdLotFrontage'])

        # Alyssa
        # fill in for masonry stuff with None and 0
        df['MasVnrType'].fillna('None',inplace=True)
        df['MasVnrArea'].fillna(0,inplace=True)
        df.loc[949,'BsmtExposure'] = 'No' # impute the ID949's BsmtExposure with the mode of 'BsmtExposure'
        #df['BsmtExposure'] = df.apply(lambda x:'No' if pd.isnull(x['BsmtExposure']) else x['BsmtExposure'], axis=1) # Generalization

        # Kisoo
        df.FireplaceQu.fillna('NA',inplace=True) # without Fireplace, there is no FireplaceQu.
        df.Electrical.fillna('SBrkr',inplace=True) # since Utility column, there is electricity obviously, so it filled with most common Electrical type 'SBrkr'

        # Wonchan
        # Feature Engineering for Time Series Columns
        #df['GarageYrBltImputed'] = np.where(df['GarageYrBlt'].isnull(), 1, 0)

        # impute the missing years with the value of the year built plus the mean of the diff of year built and garageyrbuilt
        aveDiff = round(np.mean(df['GarageYrBlt']-df['YearBuilt']))
        df['GarageYrBlt'].fillna(df['YearBuilt'] + aveDiff,inplace=True)
        #df['GarageYrBlt'].fillna(0, inplace=True)
        
        df['MSZoning'].fillna(df['MSZoning'].mode()[0],inplace=True)
        df['KitchenQual'].fillna(df['KitchenQual'].mode()[0],inplace=True)
        df['Exterior1st'].fillna(df['Exterior1st'].mode()[0],inplace=True)
        df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0],inplace=True)
        df['SaleType'].fillna(df['SaleType'].mode()[0],inplace=True)
        
        # added for test data
        df['BsmtQual'].fillna('NA',inplace=True)
        df['BsmtCond'].fillna('NA',inplace=True)
        df['BsmtFinSF1'].fillna(0,inplace=True)
        df['BsmtFinSF2'].fillna(0,inplace=True)
        df['BsmtUnfSF'].fillna(0,inplace=True)
        df['TotalBsmtSF'].fillna(0,inplace=True)
        df['BsmtFullBath'].fillna(0,inplace=True)
        df['BsmtHalfBath'].fillna(0,inplace=True)
        df['GarageCars'].fillna(0,inplace=True)
        df['GarageArea'].fillna(0,inplace=True)
        
        # fill rest with NA string       
        df = df.fillna('NA')

        return df

class Imputator(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        #df['GarageYrBltImputed'] = np.where(df['GarageYrBlt'].isnull(), 1, 0)
        #df['soldYM'] = df.apply(lambda x : str(x['YrSold'])+'0'+str(x['MoSold']) if x['MoSold']<10 else str(x['YrSold'])+str(x['MoSold']), axis=1)      
        
        df['YrSold'] = df['YrSold'].astype(str)
        df['MoSold'] = df['MoSold'].astype(str)        
        df['GarageYrBlt'] = df['GarageYrBlt'].astype(str)        
        df['MSSubClass'] = df['MSSubClass'].astype(str)
        df['BedroomAbvGr'] = df['BedroomAbvGr'].astype(str)
        df['KitchenAbvGr'] = df['KitchenAbvGr'].astype(str)
        
        ## reduce score
        '''
        df['YearBuilt'] = df['YearBuilt'].astype(str)
        df['YearRemodAdd'] = df['YearRemodAdd'].astype(str)
        df['OverallQual'] = df['OverallQual'].astype(str)
        df['OverallCond'] = df['OverallCond'].astype(str)
        df['BsmtFullBath'] = df['BsmtFullBath'].astype(str)
        df['BsmtHalfBath'] = df['BsmtHalfBath'].astype(str)
        df['FullBath'] = df['FullBath'].astype(str)
        df['HalfBath'] = df['HalfBath'].astype(str)        
        df['TotRmsAbvGrd'] = df['TotRmsAbvGrd'].astype(str)
        df['Fireplaces'] = df['Fireplaces'].astype(str)
        df['GarageCars'] = df['GarageCars'].astype(str)
        '''
        
        df['FireplaceQu'] = df.apply(lambda x:1 if x['FireplaceQu']=='Ex' else 0, axis=1).astype(str)
        df['KitchenQual'] = df.apply(lambda x:1 if x['KitchenQual']=='Ex' else 0, axis=1).astype(str)
        df['Condition2'] = df.apply(lambda x:1 if x['Condition2']=='Norm' else 0, axis=1).astype(str)
        df['Heating'] = df.apply(lambda x:1 if (x['Heating']=='GasA') | (x['Heating']=='GasW') else 0, axis=1).astype(str)
        df['HeatingQC'] = df.apply(lambda x:1 if x['HeatingQC']=='Ex' else 0, axis=1).astype(str)
        df['MiscVal'] = df.apply(lambda x:1 if x['MiscVal']>0 else 0, axis=1).astype(str)
        df['3SsnPorch'] = df.apply(lambda x:1 if x['3SsnPorch']>0 else 0, axis=1).astype(str)
        df['BsmtQual'] = df.apply(lambda x:1 if x['BsmtQual']=='Ex' else 0, axis=1).astype(str)
        df['RoofMatl'] = df.apply(lambda x:1 if x['RoofMatl']=='WdShngl' else 0, axis=1).astype(str)
        
        df['houseareaSF'] = df['BsmtFinSF1']+df['BsmtFinSF2']+df['1stFlrSF']+df['2ndFlrSF']
        df['neighbor'] = [str([i for i,n in enumerate(neighbor) if x in n][0]) for x in df.Neighborhood]
        
        ## reduce score
        '''
        df['PoolQC'] = df.apply(lambda x:1 if x['PoolQC']=='Ex' else 0, axis=1).astype(str)
        df['SaleCondition'] = df.apply(lambda x:1 if x['SaleCondition']=='Partial' else 0, axis=1).astype(str)        
        df['SaleType'] = df.apply(lambda x:1 if (x['SaleType']=='New') | (x['SaleType']=='Con') else 0, axis=1).astype(str)
        df['ExterQual'] = df.apply(lambda x:1 if x['ExterQual']=='Ex' else 0, axis=1).astype(str)
        df['PoolArea'] = df.apply(lambda x:1 if x['PoolArea']>0 else 0, axis=1).astype(str)
        df['LowQualFinSF'] = df.apply(lambda x:1 if x['LowQualFinSF']>0 else 0, axis=1).astype(str)
        df['EnclosedPorch'] = df.apply(lambda x:1 if x['EnclosedPorch']>0 else 0, axis=1).astype(str)
        df['ScreenPorch'] = df.apply(lambda x:1 if x['ScreenPorch']>0 else 0, axis=1).astype(str)
        df['porch'] = df.apply(lambda x: 1 if (x['OpenPorchSF']>0) | (x['EnclosedPorch']>0) | (x['3SsnPorch']>0) | (x['ScreenPorch']>0) else 0, axis=1).astype(str)
        '''        
        
        return df
    
class Standarizer(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        columns = [n for n in df.columns if ((df[n].dtype == int) | (df[n].dtype == float)) & (df[n].nunique()>100)]
        skewed = [columns[i] for i,v in enumerate(stats.skew(df.loc[:,columns])) if v>0.7]
        for i in skewed:
            if min(df[i])==0:
                df[i] = df[i]+1
            df[i] = special.boxcox1p(df[i], 0.15)
        
        scaler = RobustScaler()
        df['LotFrontage'] = scaler.fit_transform(df[['LotFrontage']])
        df['BsmtUnfSF'] = scaler.fit_transform(df[['BsmtUnfSF']])
        df['TotalBsmtSF'] = scaler.fit_transform(df[['TotalBsmtSF']])
        df['BsmtFinSF1'] = scaler.fit_transform(df[['BsmtFinSF1']])
        
        ## reduce score
        '''
        df['LotArea'] = scaler.fit_transform(df[['LotArea']])
        df['GrLivArea'] = scaler.fit_transform(df[['GrLivArea']])        
        '''        
        
        return df  

class OrdinalConverter(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        
        ## recude score
        '''
        ordinal = {"ExterCond": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0},
                   "ExterQual": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0},
                   "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
                   "BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
                  }
        df.replace(ordinal, inplace=True)
        '''
        
        # for pass standardization
        for i in to_numerical_columms:
            df[i] = pd.factorize(df[i])[0]
        
        return df

class DummyMaker(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        #df = pd.get_dummies(data = df, columns = onehot_categorical_columns)
        df = pd.get_dummies(df)
        
        ## reduce score
        '''
        df.drop(['Street_Grvl'],axis=1, inplace=True)        df.drop(['PoolQC_0','SaleCondition_0','SaleType_0','FireplaceQu_0','KitchenQual_0','Condition2_0','ExterQual_0','Heating_0','HeatingQC_0','porch_0','PoolArea_0','MiscVal_0','3SsnPorch_0','EnclosedPorch_0','ScreenPorch_0','BsmtQual_0','RoofMatl_0'], axis=1, inplace=True)
        '''
        return df
    
class Featuredropper(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):                
        df.drop(['BsmtFinSF2','2ndFlrSF','LowQualFinSF','OpenPorchSF'], axis=1, inplace=True)
        
        ## reduce score
        '''
        df.drop(n_drop_feature, axis=1, inplace=True)
        df.drop(c_drop_feature, axis=1, inplace=True)
        df.drop(custom_categorical_columns, axis=1, inplace=True)
        '''
                                  
        return df   
    
def pre_processing(train_df, result_df, test_df):
    pre_pipeline = Pipeline([
        ('nullFiller', NullFiller()),
        ('imputator', Imputator()),
        ('standarizer', Standarizer()),
        ('ordinalConverter', OrdinalConverter()),
        ('featuredropper', Featuredropper()),        
        ('dummyMaker', DummyMaker()),                
    ])
    
    # remove outliers(by EDA)
    outliers = [('LotFrontage',300),('BsmtFinSF1',5000),('TotalBsmtSF',6000),('1stFlrSF',4000),('GrLivArea',4600)] #,('GarageArea',1220), ('LotArea',150000)
    for i in outliers:
        idx = train_df[train_df[i[0]]>i[1]].index
        train_df.drop(idx, axis=0, inplace=True)
        result_df.drop(idx, axis=0, inplace=True)
    
    # merge train and test data set
    train_length = len(train_df)    
    all_df = pd.concat([train_df, test_df], axis=0)    
    all_df = all_df.reset_index(drop=True)
    all_df = pre_pipeline.transform(all_df)
    train_df = all_df.iloc[:train_length]
    test_df = all_df.iloc[train_length:]
    result_df = np.log1p(result_df)
    
    train_df.reset_index(drop=True)
    result_df.reset_index(drop=True)
    test_df.reset_index(drop=True)
    
    return train_df, result_df, test_df