import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, TransformerMixin
from scipy.special import boxcox1p

#categorical_columns = ['Neighborhood','MSZoning','Street','LotShape','LotConfig','Condition2','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','Heating','CentralAir','Electrical','KitchenQual','FireplaceQu','GarageType','GarageFinish','GarageQual','SaleType','SaleCondition','GarageYrBltImputed']

categorical_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities','LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2','BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st','Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual','Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt','GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC','Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

class NullFiller(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        #print(df.isnull().sum().sort_values(ascending = False))
        #print(df.shape)
        # Hyelee
        ## Alley : convert NaN => NA
        df['Alley'].fillna('NA',inplace=True)
        ## LotFrontage : convert NaN => ???? 
        #<CASE2> Using Neighborhood 
        df_tmp = df[~df.LotFrontage.isnull()]
        df_md_lotfrontage = df_tmp.groupby('Neighborhood').agg('median')[{'LotFrontage'}]
        df_md_lotfrontage = df_md_lotfrontage.rename(columns={"LotFrontage" : "mdLotFrontage"})
        df_tmp = df_tmp.merge(df_md_lotfrontage, how="inner", on='Neighborhood')
        df = pd.concat([df.reset_index().set_index('Neighborhood'),df_md_lotfrontage], axis=1, join='inner').reset_index().set_index('Id')
        df['LotFrontage'].fillna(df['mdLotFrontage'],inplace=True)
        df = df.drop(columns=['mdLotFrontage'])
        #df['LotFrontage'].fillna(df['LotArea'] *0.007206024910841549,inplace=True)

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
        df['GarageYrBltImputed'] = np.where(df['GarageYrBlt'].isnull(), 1, 0)
        df['RemodAddNew'] = np.where((df['YearRemodAdd']-df['YearBuilt'] >0), 1, 0)
        df["bathMerge"] = df.BsmtFullBath * 1 + df.BsmtHalfBath * 0.5 + df.FullBath * 1 + df.HalfBath * 0.5
        df["porchMerge"] = df.WoodDeckSF + df.OpenPorchSF + df.EnclosedPorch + df['3SsnPorch'] + df.ScreenPorch
        # Divide 3 group "Neighborhood" 
        group_negh_1 = ['Blmngtn','StoneBr', 'Somerst', 'NridgHt', 'NoRidge', 'Timber', 'Veenker', 'ClearCr', 'Crawfor', 'CollgCr']
        group_negh_2 = ['SawyerW', 'NWAmes', 'NPkVill', 'NAmes', 'Gilbert', 'Mitchel']
        group_negh_3 = ['MeadowV','IDOTRR','BrkSide','OldTown','SWISU','Sawyer','BrDale','Blueste', 'Edwards']

        df['NeighborhoodNew'] = df["Neighborhood"].map(lambda x: 1 if(x in group_negh_1) else(2 if(x in group_negh_2) else 3))

        return df
    
class Standarizer(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        ## Standardization 
        
        # lambda = 0  =>  y = log(1+x)
        # y = boxcox1p(x, 0)
      
        # lambda = 1  =>  y = x
        # y = boxcox1p(x, 1)
       
        # lambda = 2  =>  y = 0.5*((1+x)**2 - 1) = 0.5*x*(2 + x)
        # y = y = boxcox1p(x, 2)
        
        ## skew > 9 || skew < -0.5 
        df['MiscVal'] = boxcox1p(df['MiscVal'],0)
        df['GrLivArea'] = boxcox1p(df['GrLivArea'],0)
        df['LotFrontage'] = boxcox1p(df['LotFrontage'],0)
        df['LotArea'] = boxcox1p(df['LotArea'],0)
        df['LowQualFinSF'] = boxcox1p(df['LowQualFinSF'],0) 
        df['3SsnPorch'] = boxcox1p(df['3SsnPorch'],0)
        df['GarageYrBlt'] = boxcox1p(df['GarageYrBlt'],0)
        df['YearBuilt'] = boxcox1p(df['YearBuilt'],0)
    
        ## Normalization 
        scaler = StandardScaler()
        #df['BsmtUnfSF'] = scaler.fit_transform(df[['BsmtUnfSF']])
        #df['TotalBsmtSF'] = scaler.fit_transform(df[['TotalBsmtSF']])
        #df['BsmtFinSF1'] = scaler.fit_transform(df[['BsmtFinSF1']])
        return df  

class OrdinalConverter(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        for i in categorical_columns:
            df[i] = pd.factorize(df[i])[0]+1  
        return df

class DummyMaker(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        df = pd.get_dummies(data=df, columns=categorical_columns)        
        return df
    
class Featuredropper(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        # multicollinearity , and merge column 
        n_drop_feature = ['GarageArea', 'BsmtHalfBath','BsmtFullBath','HalfBath','FullBath','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']
        
        # categorical variable 
        c_drop_feature=['PoolQC','Alley']
        #c_drop_feature = ,['MSSubClass','Alley','LandContour','Utilities','LandSlope','Condition1','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','BsmtFinType1','BsmtFinType2','HeatingQC','Functional','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature']
        #custom_drop_feature = ['GarageYrBltImputed']
        
        df.drop(n_drop_feature, axis=1, inplace=True)
        df.drop(c_drop_feature, axis=1, inplace=True)
        #df.drop(custom_drop_feature, axis=1, inplace=True)
        
        ## Edit dummy variable column
        for i in range(len(c_drop_feature)):
            categorical_columns.remove(c_drop_feature[i])
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
    
    ## delete outlier(Grid Area, OverallQual)
    outliers = [('GrLivArea', 4500)]
    for i in outliers:
        idx = train_df[train_df[i[0]] > i[1]].index
        print('idx:  ', idx)
        train_df.drop(idx, inplace=True)
        result_df.drop(idx, inplace=True)
    
    print("train_df: " , train_df.shape)
    print("test_df: " , test_df.shape)
    train_length = len(train_df)
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df = pre_pipeline.transform(all_df)
    train_df = all_df.iloc[:train_length]
    test_df = all_df.iloc[train_length:]
    result_df = np.log1p(result_df)
    
    return train_df, result_df, test_df