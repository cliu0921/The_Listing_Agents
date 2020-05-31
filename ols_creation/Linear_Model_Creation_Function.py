def OLS_Model_Creation(path):
    '''
    This function takes in Aimes, Iowa housing data and spits out a scikit multi-linear regression model
    '''
    
    # import relevant libraries
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression 
    from sklearn.model_selection import train_test_split
    
    # read data in from parent directory
    from read_path_module import read_data_relative_path
    try:
        df_train = read_data_relative_path(relative_dataset_path = path, data_type='csv')
    except:
        df_train = path
    
    # impute truly "missing" data (i.e. the NA's do not have significance)
    df_train['LotFrontage'] = df_train['LotFrontage'].mask(df_train['LotFrontage'].isnull(), np.random.uniform(df_train['LotFrontage'].min(), df_train['LotFrontage'].max(), size = df_train['LotFrontage'].shape))
    df_train['GarageYrBlt'] = df_train['GarageYrBlt'].mask(df_train['GarageYrBlt'].isnull(), np.random.uniform(df_train['GarageYrBlt'].min(), df_train['GarageYrBlt'].max(), size = df_train['GarageYrBlt'].shape))
    df_train['MasVnrArea'] = df_train['MasVnrArea'].mask(df_train['MasVnrArea'].isnull(), np.random.uniform(df_train['MasVnrArea'].min(), df_train['MasVnrArea'].max(), size = df_train['MasVnrArea'].shape))

    # fill rest of NA's with Nothing to add categorical meaning (i.e. a NA in poolQC means that there is no pool)
    df_train.fillna('Nothing', inplace = True)  
    
    # drop non-needed columns
    df = df_train.drop(['Id'], axis = 1)
    df = df.drop(['Unnamed: 0'], axis = 1) 
    
    # create df copy and isolate the categorical columns for dummification
    categorical = ['Alley', 'BldgType_group', 'BsmtCond_group', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual_group', 'CentralAir', 'Condition1_group', 'Electrical_group', 'ExterCond_group', 'ExterQual', 'Exterior1st_group', 'Exterior2nd_group', 'Fence', 'FireplaceQu', 'Foundation_group', 'GarageCond_group', 'GarageFinish', 'GarageQual', 'GarageType', 'HeatingQC_group', 'HouseStyle_group', 'KitchenQual', 'LandContour_group', 'LandSlope', 'LotConfig_group', 'LotShape_group', 'MS_Zoning_group', 'MasVnrType_group', 'Neighborhood', 'PavedDrive', 'PoolQC', 'RoofStyle_group', 'SaleCondition_group', 'SaleType_group']
    df_1 = df[categorical]
    df_dum = pd.get_dummies(df_1, drop_first = True)  
    
    # create df copy of numerical variables and concatenate this with the dummified df
    df_num = df[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']] 
    df = pd.concat([df_num, df_dum], axis = 1) 
    
    # Filter out outliers
    mult_upper = 4
    mult_lower = 1.5
    med = df['SalePrice'].median()
    mean = df['SalePrice'].mean()
    std = df['SalePrice'].std()
    df = df.loc[(df['SalePrice'] > med - (mult_lower * std) ) & (df['SalePrice'] < med + (mult_upper * std))]
    
    # create X and y      
    X = df.drop(['SalePrice'], axis = 1)
    y = np.log(df['SalePrice'])
    
    # train_test_split   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)  
    
    # linear regression   
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    
    # fit model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    
    return lin_reg, X_train, X_test, y_train, y_test