from FLUCCOplus.scenarios import EXCEL_TO_EM_colnames
from FLUCCOplus.utils import *

@logg
def residual_load(df, load, sources):
    return df

@logg
def scale_to_scenario(df, factors):
    """takes a EM dataframe and scaling factors and returns a dict with keys [scenario]:  vals [scaled dataframes]"""
    scens = dict()
    columns = sorted(list(EXCEL_TO_EM_colnames.keys()))
    for i, sc in enumerate(factors.index):
        scens[sc] = df[columns].copy()
        for col in columns:
            f = factors[col][i]
            year = factors["EM"][i]
            scens[sc].loc[scens[sc].index.year == year, col] = df.loc[df.index.year == year, col] * f

        scens[sc]["RES0 (Bedarf-PV,Wind,Laufkraft)"] = scens[sc]["Strombedarf"] - scens[sc]["Volatile EE"]
        scens[sc]["RES1 (RES0-Pumpspeicher)"] = scens[sc]["RES0 (Bedarf-PV,Wind,Laufkraft)"] - scens[sc][
            "Pumpspeicher"]
        scens[sc]["RES2 (RES1-Nicht-Volatile)"] = scens[sc]["RES1 (RES0-Pumpspeicher)"] - scens[sc][
            "Nicht-Volatile"]
    return scens