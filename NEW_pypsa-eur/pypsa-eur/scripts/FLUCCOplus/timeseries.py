from FLUCCOplus.config import *


@logg
def remove_outliers(df, std=3):
    # keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
    return df[df.apply(lambda x: np.abs(x - x.mean()) / x.std() < std).all(axis=1)]

@logg
def trim_outliers_to_max(df):
    # df.column_name.loc[df.column_name > max_value] = max_value
    pass

@logg
def normalize(df):
    return df.apply(lambda x: x/x.max())



@log
def save(df, path):
    """return df.to_csv(path,
              sep=";",
              decimal=",",
              encoding="cp850")"""
    return df.to_csv(path,
              sep=";",
              decimal=",",
              encoding="cp850")