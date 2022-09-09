
def clean_for_parquet(df):
    """Convert all string types to non-bytes strings and replace np.nan with NA string
    
    Makes pandas.to_parquet happy. If there are some columns with both ints and other datatypes and parquet can't handle it
    """
    cleaned_for_parquet = df.copy()
    for col, dtype in df.dtypes.iteritems():
        if dtype == 'object':
            cleaned_for_parquet[col] = cleaned_for_parquet[col].fillna("NA").astype(str)
    return cleaned_for_parquet
