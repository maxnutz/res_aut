
import FLUCCOplus.config as config
from FLUCCOplus.utils import *

pp_carriers = [
    'Laufkraftwerke',
    'Speicherkraftwerke',
    'FossileBrennstoffeundDerivate',
    'BiogeneBrennstoffe',
    'Windkraftwerke',
    'Photovoltaik',
    'Geothermie',
    'StatistischeDifferenz',
    'PhysikalischeImporte']
pp_cats = [
    'Wasserkraft',
    'Wärmekraftwerke',
    'WindkraftwerkePhotovoltaikGeothermie']
pp_sum = [
    'Bruttoerzeugung']
pc_sum = [
    'AufbringungVerwendung']
pc_cats = [
    'Inlandstromverbrauch',
    'Inlandstromverbrauch3']
pc_uses = [
    'Endverbrauch1',
    'VerbrauchfürPumpspeicher',
    'KWeigenbedarf',
    'Netzverluste',
    'PhysikalischeExporte']

EC_COLUMNS = pp_carriers + pp_cats + pp_sum + pc_sum + pc_cats + pc_uses



@logg
def start_pipeline(df):
    return df.copy()

@logg
def parse_econtrol(path):
    return pd.read_excel(path,
                       sheet_name="FLUCCOplus",
                       skiprows=[1],
                       index_col=0)


@logg
def clean_colnames(df):
    clean_colnames = ["".join([c for c in col if c not in "()\n- ,="]) for col in df.columns]
    dfreturn = df.rename(columns=dict(zip(df.columns, clean_colnames)))
    return dfreturn

@logg
def plot_stacked(df):
    df.plot.bar(stacked=True)
    plt.show()
    return df

@logg
def production(df):
    return df

