
import FLUCCOplus.config as config
from FLUCCOplus.utils import *



@log
def read(path):
    from FLUCCOplus.config import DATA_RAW
    return pd.read_excel(DATA_RAW / "szenarien"/ path,
                  sheet_name="scenarios",
                  index_col=0, skiprows=range(1, 3))

@logg
def start_pipeline(df):
    return df.copy()

@logg
def rename_cols_to_common(df):
    df = df.rename(columns=EM_TO_EXCEL_colnames)
    return df



@logg
def convert_PJ_to_GWH(df):
    jahr = df["Jahr"]
    df = df.drop("Jahr", axis=1)
    df = df * PJ_TO_GWH
    df.insert(0,"Jahr", jahr)
    df.index.rename("Szenario", inplace=True)
    return df

@logg
def NaNtoZero(df):
    return df.fillna(0)

@logg
def format_df(df):
    return df.astype({"Jahr":"int32"})


@logg
def get_scenario(scenario_name, col_dict, Excel_to_EM_dict):
    """gets the preprocessed scenario including residual loads"""
    df_scenario = pd.DataFrame()

    for col in Excel_to_EM_dict.keys():
        # %%
        c = col_dict[col]
        df_scenario[col] = c[scenario_name]

    df_scenario["RES0 (Bedarf-PV,Wind,Laufkraft)"] = df_scenario["Strombedarf"] - df_scenario["Volatile EE"]
    df_scenario["RES1 (RES0-Pumpspeicher)"] = df_scenario["RES0 (Bedarf-PV,Wind,Laufkraft)"] - df_scenario[
        "Pumpspeicher"]
    df_scenario["RES2 (RES1-Nicht-Volatile)"] = df_scenario["RES1 (RES0-Pumpspeicher)"] - df_scenario["Nicht-Volatile"]

    # df_sc["RES0"][scenario_name] = df_scenario["RES0"]
    return df_scenario



def all():
    """
    Returns all W2S Scenarios in [GWh/a]
    :return:
    """
    sc = (read("szenarien_w2s.xlsx")
          .pipe(start_pipeline)
          .pipe(NaNtoZero)
          .pipe(format_df)
          .pipe(convert_PJ_to_GWH)
          )
    return sc


@log
def factors(source, target, scenarios):
    """

    :param source: String or Index int of source scenario
    :param target: String or Index int of target scenario
    :param scenarios: Dataframe with Scenarios, must include all EM_TO_EXCEL_colnames
    :return: dict with factors for all EM_TO_EXCEL_colnames
    """
    if type(source) == int:
        src = scenarios.index[source]
    else: src = source

    if type(target) == int:
        tgt = scenarios.index[target]
    else: tgt = target

    carriers = EM_TO_EXCEL_colnames.values()
    f = scenarios.loc[tgt, carriers] / scenarios.loc[src, carriers]
    factors = {"source": src, "target": tgt}
    factors.update({i: j for i, j in zip(carriers, f)})

    return factors


class Scenario:


    def __init__(self, name, scenario, em_base=None):
        self.name = name
        scenarios = all()
        self.annuals = scenarios[scenarios.index==name].transpose().rename(columns={self.name: "target"})

        self.scalable = ["Strombedarf","Pumpspeicher", "Laufkraft", "Windkraft", "Photovoltaik"]
        self.excel_em_prods =  {EXCEL_TO_EM_colnames[k]:k for k in self.scalable}
        self.excel_em_cons = {k.replace("production", "consumption"):v for k,v in self.excel_em_prods.items()}

        self.base_df = pd.DataFrame()
        self.base_year = None
        self.load_base(em_base)

        self.TSD = pd.DataFrame()
        self._scale_base_to_target()


    def load_base(self, em_base):
        """
        loads an electricity map year as .em_base
        :param em_base: a year of electricity map data as dict with {year, df}
        :return:
        """
        self.base_df = em_base["df"]
        self.base_year = em_base["year"]
        self.annuals["base"] = em_base["df"][self.em_excel.keys()].rename(columns={k: v for k, v in zip(self.em_excel, self.scalable)}).sum() / 1000
        self.annuals.loc["Jahr", "base"] = self.base_year
        self._calc_scale()

    def _calc_scale(self):
        for scale in self.scalable:
            self.annuals.loc[scale, "scale"] = self.annuals.loc[scale,"target"] / self.annuals.loc[scale,"base"]

    def _scale_base_to_target(self):
        for scale in self.scalable:
            f = self.annuals.loc[scale, "scale"]
            self.TSD[scale] = f * self.base_df[self.excel_em[scale]]

    @property
    def em_excel(self):
        if self.base_year >= 2018:
            return self.excel_em_prods
        elif 2015 <= self.base_year <= 2017:
            return self.excel_em_cons

    @property
    def excel_em(self):
        return {v: k for k,v in self.em_excel.items()}

    def plot(self):
        self.TSD.plot(color=config.COLORS)

    def __repr__(self):
        repr = ""
        repr += self.name + "\n" * 2
        repr += str(self.annuals)
        return repr

    def __str__(self):
        return self.repr()

    def scaling_targets(self):
        pass

