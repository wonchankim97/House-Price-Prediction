import pandas as pd
import numpy as np
from scipy import stats
from sklearn.pipeline import Pipeline, TransformerMixin

categorical_columns = ['Neighborhood','MSZoning','Street','LotShape','LotConfig','Condition2','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','Heating','CentralAir','Electrical','KitchenQual','FireplaceQu','GarageType','GarageFinish','GarageQual','SaleType','SaleCondition','GarageYrBltImputed']

class NullFiller(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        # Hyelee
        ## Alley : convert NaN => NA
        df['Alley'].fillna('NA',inplace=True)
        ## LotFrontage : convert NaN => ???? 
        # <CASE2> Using Neighborhood 
        df_tmp = df[~df.LotFrontage.isnull()]
        df_md_lotfrontage = df_tmp.groupby('Neighborhood').agg('median')[{'LotFrontage'}]
        df_md_lotfrontage = df_md_lotfrontage.rename(columns={"LotFrontage" : "mdLotFrontage"})
        df_tmp = df_tmp.merge(df_md_lotfrontage, how="inner", on='Neighborhood')
        df = pd.concat([df.reset_index().set_index('Neighborhood'),df_md_lotfrontage], axis=1, join='inner').reset_index().set_index('Id')
        df['LotFrontage'].fillna(df['mdLotFrontage'],inplace=True)
        df = df.drop(columns=['mdLotFrontage'])

        # Alyssa
        # fill in for masonry stuff with None and 0
        df['MasVnrType'].fillna('None',inplace=True)
        df['MasVnrArea'].fillna(0,inplace=True)
        #df.loc[949,'BsmtExposure'] = 'No' # impute the ID949's BsmtExposure with the mode of 'BsmtExposure'
        df['BsmtExposure'] = df.apply(lambda x:'No' if pd.isnull(x['BsmtExposure']) else x['BsmtExposure'], axis=1) # Generalization

        # Kisoo
        df.FireplaceQu.fillna('NA',inplace=True) # without Fireplace, there is no FireplaceQu.
        df.Electrical.fillna('SBrkr',inplace=True) # since Utility column, there is electricity obviously, so it filled with most common Electrical type 'SBrkr'

        # Wonchan
        # Feature Engineering for Time Series Columns
        df['GarageYrBltImputed'] = np.where(df['GarageYrBlt'].isnull(), 1, 0)

        # impute the missing years with the value of the year built plus the mean of the diff of year built and garageyrbuilt
        aveDiff = np.mean(df['GarageYrBlt']-df['YearBuilt'])
        df['GarageYrBlt'].fillna(df['YearBuilt'] + aveDiff,inplace=True)
        
        # added for test data
        df['BsmtQual'].fillna('NA',inplace=True)
        df['BsmtCond'].fillna('NA',inplace=True)
        df['BsmtFinSF1'].fillna(0,inplace=True)
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
        
        return df
    
class Standarizer(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        df['1stFlrSF'] = stats.boxcox(df['1stFlrSF'])[0]
        df['GrLivArea'] = stats.boxcox(df['GrLivArea'])[0]
        df['LotArea'] = stats.boxcox(df['LotArea'])[0]
        df['LotFrontage'] = stats.boxcox(df['LotFrontage'])[0]  
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
        pd.get_dummies(data=df, columns=categorical_columns)        
        return df
    
class Featuredropper(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        # numeric variable corr<0.20
        n_drop_feature = ['BsmtFinSF2','LowQualFinSF','EnclosedPorch','3SsnPorch','PoolArea','MiscVal']
        # categorical variable chi^2 p-value >0.05
        c_drop_feature = ['MSSubClass','Alley','LandContour','Utilities','LandSlope','Condition1','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','BsmtFinType1','BsmtFinType2','HeatingQC','Functional','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature']
        df.drop(n_drop_feature, axis=1, inplace=True)
        df.drop(c_drop_feature, axis=1, inplace=True)
        return df    

def pre_processing(train_df, test_df):
    pre_pipeline = Pipeline([
        ('nullFiller', NullFiller()),
        ('imputator', Imputator()),
        ('standarizer', Standarizer()),
        ('ordinalConverter', OrdinalConverter()),
        ('dummyMaker', DummyMaker()),
        ('featuredropper', Featuredropper()),
    ])
    
    train_length = len(train_df)
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df = pre_pipeline.transform(all_df)
    train_df = all_df.iloc[:train_length]
    test_df = all_df.iloc[train_length:]
    
    return train_df, test_df