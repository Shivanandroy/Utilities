#############

def calculate_vif(DataFrame, thresh=5.0):
    '''
    DataFrame = Input Dataframe
    thresh = Maximum threshfold for Variance Inflation Factor (VIF)
    
    returns:
    
    1. Filtered DataFrame with columns where VIF < threshold
    2. VIF Score DataFrame
    
    
    
    '''
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    X = DataFrame.select_dtypes(include=[np.number])
    
    
    variables = list(range(X.shape[1]))
    dropped=True
    while dropped:
        dropped=False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix) for ix in range(X.iloc[:,variables].shape[1])]
        #print('\n')
        #print(vif)
        
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            #print('Dropping \'' + X.iloc[:,variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped=True
    
    #print('\n')   
    return X.iloc[:,variables], pd.DataFrame(vif, index=X.columns[variables].tolist(), columns=['VIF'])

################################################### Correlation #############################################################
corr_matrix = df.corr().abs()

#the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
os = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False)).reset_index()
