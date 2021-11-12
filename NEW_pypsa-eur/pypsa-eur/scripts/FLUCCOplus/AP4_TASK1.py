#%%

import pandas as pd

E_CONTROL_XLSX = "../data/e-control/BStGes-JR1_Bilanz.xlsx"

ec_raw = pd.read_excel(E_CONTROL_XLSX, sheet_name="FLUCCOplus", skiprows=[1], index_col=0)

#%%
from FLUCCOplus import econtrol as ec

ec_clean = (ec_raw
            .pipe(ec.start_pipeline)
            .pipe(ec.clean_colnames)
            )

ec_production = (ec_clean
                 .pipe(ec.production)
                 )

#%%
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
pp_sum = ['Bruttoerzeugung']
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
    'PhysikalischeExporte', ]

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.tight_layout()
ec_clean[pp_carriers].plot.bar(stacked=True, ax=ax1)
ec_clean[pc_uses].iloc[::-1].plot.bar(stacked=True, ax=ax2)
plt.show()

#%%
ec_clean.head()

#%%


