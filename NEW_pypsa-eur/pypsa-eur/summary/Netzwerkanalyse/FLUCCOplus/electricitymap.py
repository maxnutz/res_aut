
from FLUCCOplus.utils import *
import FLUCCOplus.config as config


header_junk = ['production_sources', 'Jahr', 'monat', 'Tag', 'Stunde', 'Datum', 'Tag des Jahres',
               'Tag des Monats', 'Uhrzeit', 'timestamp', 'zone_name']

# Columns of electricity Map 2018-19 and 15-17
common = [
    'total_export_avg',
    'total_import_avg',
    'carbon_intensity_avg',
    'total_consumption_avg',
    'total_production_avg']

# df 15-17
power_origins = [
    'power_origin_percent_biomass_avg',
    'power_origin_percent_coal_avg',
    'power_origin_percent_gas_avg',
    'power_origin_percent_hydro_avg',
    'power_origin_percent_nuclear_avg',
    'power_origin_percent_oil_avg',
    'power_origin_percent_solar_avg',
    'power_origin_percent_wind_avg',
    'power_origin_percent_geothermal_avg',
    'power_origin_percent_unknown_avg',
    'power_origin_percent_hydro_discharge_avg']

# df 18-19 only
totals = [
    'total_production_avg',
    'total_storage_avg',
    'total_discharge_avg',
    'total_import_avg',
    'total_export_avg',
    'total_consumption_avg']

power_net_imports = ['power_net_import_CH_avg', 'power_net_import_CZ_avg',
                     'power_net_import_DE_avg', 'power_net_import_HU_avg',
                     'power_net_import_IT-NO_avg', 'power_net_import_SI_avg']

production = [
    'power_production_biomass_avg',
    'power_production_coal_avg',
    'power_production_gas_avg',
    'power_production_hydro_avg',
    'power_production_nuclear_avg',
    'power_production_oil_avg',
    'power_production_solar_avg',
    'power_production_wind_avg',
    'power_production_geothermal_avg',
    'power_production_unknown_avg', ]
consumption = [
    'power_consumption_biomass_avg', 'power_consumption_coal_avg',
    'power_consumption_gas_avg', 'power_consumption_hydro_avg',
    'power_consumption_nuclear_avg', 'power_consumption_oil_avg',
    'power_consumption_solar_avg', 'power_consumption_wind_avg',
    'power_consumption_geothermal_avg', 'power_consumption_unknown_avg',
    'power_consumption_battery_discharge_avg',
    'power_consumption_hydro_discharge_avg']

carbon_intensities = [
    'carbon_intensity_avg',
    'carbon_intensity_production_avg',
    'carbon_intensity_discharge_avg',
    'carbon_intensity_import_avg']
carbon_rate = ['carbon_rate_avg']
carbon_origins = ['carbon_origin_percent_biomass_avg', 'carbon_origin_percent_coal_avg',
                  'carbon_origin_percent_gas_avg', 'carbon_origin_percent_hydro_avg',
                  'carbon_origin_percent_nuclear_avg', 'carbon_origin_percent_oil_avg',
                  'carbon_origin_percent_solar_avg', 'carbon_origin_percent_wind_avg',
                  'carbon_origin_percent_geothermal_avg',
                  'carbon_origin_percent_unknown_avg',
                  'carbon_origin_percent_battery_discharge_avg',
                  'carbon_origin_percent_hydro_discharge_avg', ]
latest_forecasted = [
    'latest_forecasted_dewpoint_avg',
    'latest_forecasted_precipitation_avg',
    'latest_forecasted_solar_avg',
    'latest_forecasted_temperature_avg',
    'latest_forecasted_wind_x_avg',
    'latest_forecasted_wind_y_avg',
    'latest_forecasted_price_avg',
    'latest_forecasted_production_avg',
    'latest_forecasted_consumption_avg',
    'latest_forecasted_power_net_import_CH_avg',
    'latest_forecasted_power_net_import_CZ_avg',
    'latest_forecasted_power_net_import_DE_avg',
    'latest_forecasted_power_net_import_HU_avg',
    'latest_forecasted_power_net_import_IT-NO_avg',
    'latest_forecasted_power_net_import_SI_avg',
    'latest_forecasted_production_solar_avg',
    'latest_forecasted_production_wind_avg']
power_net_discharge = ['power_net_discharge_battery_avg', 'power_net_discharge_hydro_avg']

# aggregate
vars = totals + carbon_intensities

renewables = ["biomass",
              "hydro",
              "solar",
              "wind",
              "geothermal"]

fossile = ["coal",
           "gas",
           "nuclear",
           "oil",
           "unknown"]

discharge = ["battery_discharge",
             "hydro_discharge"]

carriers = renewables + fossile
sources = renewables + fossile + discharge

def col(prefix, vars, suffix="_avg"):
    """what i want: give variable (carrier) list with prefix"""
    # return valid col list
    if type(vars) is str:
        return ''.join([prefix, "_", vars, suffix])
    else:
        return [''.join([prefix, "_", var, suffix]) for var in vars]

pp = "power_production"
pps_RE = col(pp, renewables)
pps_NRE = col(pp, fossile)  # no discharge

pc = "power_consumption"
pcs_RE = col(pc, renewables)
pcs_NRE = col(pc, fossile)
pcs_discharge = col(pc, discharge)

cop = "carbon_origin_percent"
cops_RE = col(cop, renewables)
cops_NRE = col(cop, fossile)
cops_discharge = col(cop, discharge)

pps = pps_RE + pps_NRE
pcs_sources = pcs_RE + pcs_NRE
pcs = pcs_sources + pcs_discharge
cops = cops_RE + cops_NRE + cops_discharge

carrier_colors = {
    "biomass": "xkcd:green",
    "hydro": "xkcd:aqua",
    "solar": "yellow",
    "wind": "xkcd:azure",
    "geothermal": "brown",
    "coal": "darkgray",
    "gas": "xkcd:navy",
    "nuclear": "magenta",
    "oil": "black",
    "unknown": "gray",
    "battery_discharge": "orange",
    "hydro_discharge": "blue"}


@log
def read_raw(file):
    """
    reads a raw electricity map .csv and returns the df
    """
    return pd.read_csv(config.DATA_RAW / "electricityMap" / file,
                     delimiter=";",
                     parse_dates=["datetime"],
                     index_col="datetime")

@log
def read_interim(file, encoding="cp850"):
    return pd.read_csv(config.DATA_INTERIM / file,
                       delimiter=";",
                       parse_dates=["datetime"],
                       index_col="datetime",
                       decimal=",",
                       encoding=encoding
                       )


@logg
def start_pipeline(df):
    return df.copy()

@logg
def clean151617(df):
    """returns a clean em df 15-17"""
    return (df
            .drop("local_datetime", axis=1)
            .drop
            (
                'datetime,local_datetime,total_production_avg,total_consumption_avg,'
                'total_import_avg,total_export_avg,'
                'carbon_intensity_avg,power_origin_percent_biomass_avg,power_origin_percent_coal_avg,power_origin_percent_gas_avg,power_origin_percent_hydro_avg,power_origin_percent_nuclear_avg,power_origin_percent_oil_avg,power_origin_percent_solar_avg,power_origin_percent_wind_avg,power_origin_percent_geothermal_avg,power_origin_percent_unknown_avg,'
                'power_origin_percent_hydro_discharge_avg', axis=1)
            .astype(float)
            )



@log
def split_into_years(df):
    """separate dataframe in dict(year: df[year]"""
    years = df.index.year.unique().values
    # print(f"{len(years)} Jahre: ", years)
    # #%%
    # for i, year in enumerate(years):
    df_dict = {year: df_slice for year, df_slice in zip(years, [df[df.index.year == y] for y in years])}
    return df_dict


@logg
def year(df, year: int):
    return df[df.index.year == year]


@logg
def as_df(dict: dict):
    df = pd.concat(dict.values())
    return df

@logg
def calc_power_consumption_from_percent(df):
    cols = carriers +["hydro_discharge"]
    for c in cols: #keine discharge (battery, hydro)
        # tc = df["total_consumption_avg"].sum()
        pc = col("power_consumption", c)
        percent = col("power_origin_percent", c)
        df[pc] = df[percent] / 100 * df["total_consumption_avg"]  #wegen%
    df[col("power_consumption", "battery_discharge")] = 0
    return df

@logg
def calc_aggregates(df):
    df["Importe"] = df["total_import_avg"] - df["total_export_avg"]
    df["Wasserkraft"] = df["Laufkraft"] + df["Pumpspeicher"]
    df["Volatile EE"] = df["Laufkraft"] + df["Windkraft"] + df["Photovoltaik"]
    df["Nicht-Volatile"] = df["Stromproduktion"] - df["Wasserkraft"] - df["Windkraft"] - df["Photovoltaik"]
    return df


@log
def fetch_1819():
    """
    returns the cleaned elmap dataset from 2018 and 2019:
    :return:
    """
    return (read_raw("Electricity_map_CO2_AT_2018_2019.csv")
            .pipe(start_pipeline)
            .drop(header_junk, axis=1)
            .astype(float)
            )

@log
def fetch_151617():
    """
    returns the cleaned elmap dataset from 2015-2017: but there is alot of dropping
    :return:
    """
    return (read_raw("Electricity_map_CO2_AT_2015_2017.csv")
            .pipe(start_pipeline)
            .pipe(clean151617)
            .pipe(calc_power_consumption_from_percent)
            )



@log
def preprocess():
    """
    returns a df with all columns common to elmap1819 and elmap15-17
    """
    em18 = fetch_1819()

    em15 = fetch_151617()

    em18set = set(em18.columns)
    em15set = set(em15.columns)
    common_cols = sorted(list(em15set & em18set))

    em = pd.concat([em15[common_cols], em18[common_cols]])
    return em


@log
def fetch_common():
    return read_interim("em_common_15-19.csv")

@log
def fetch(year=None, common=False):
    if year:
        if year >= 2018:
            df = fetch_1819()
        elif 2015 <= year <= 2017:
            df = fetch_151617()
        return df[df.index.year == year]

    elif common:
        return fetch_common()

    else:
        raise AttributeError("Need to specify a parameter: year or common=True!")

@logg
def drop_suffixes(df):
    print("not implemented yet")
    return df


@logg
def save_to_csv(df_dict: dict, scenario_folder="data/scenarios/"):
    from datetime import datetime
    date = datetime.isoformat(datetime.now())[:-10].replace(":", "-")
    folder = scenario_folder + date
    import os
    os.mkdir(folder)
    for name, df in df_dict.items():
        df.to_csv("".join([folder, "/", name, "MW.csv"]),
                  sep=";",
                  decimal=",",
                  encoding="cp850")

        print(name, " saved!")


def get(year, column, set_name=None):
    """returns a given column of a given year as a series, optionally renamed"""
    if year in [2015,2016,2017]:
        df = fetch_151617()
    elif year in [2018, 2019]:
        df = fetch_1819()
    else:
        raise ValueError(f"Year must be in [2015-2019] as int, got {year}")
    series = df[df.index.year == year][column]
    if set_name:
        series.rename(set_name, inplace=True)
    return series

def annual_emissions():
    """returns a series of electricity map co2 averages based on the resampled dataframe"""
    df = read_interim("em_common_15-19.csv")

    avg = pd.Series([df.loc[df.index.year == y, "carbon_intensity_avg"].mean() for y in df.index.year.unique()], index=[df.index.year.unique()])
    avg.index = avg.index.map(str)

    avg["2015-2018"] = df.loc[df.index.year!=2019, "carbon_intensity_avg"].mean()
    avg["2015-2019"] = df[ "carbon_intensity_avg"].mean()
    avg["OIB 2019 (Entso-E 2014-2018)"] = 227

    return avg


def plot_PE_factors():
    pd.DataFrame(PE_factors_OIB2019) \
        .drop(["fPE"]) \
        .transpose() \
        .plot(kind="bar", color=["grey", "lightgreen"], stacked=True)


def primary_energy(df, type="fPE"):
    """
    returns the primary energy by multiplying a df with _sources_
    with the utils.PE_factors_OIB2019

    Conversion factors are taken from the OIB RL 6 (2019).

    *Hydro, Solar and Wind* were assumed with $f_{PE,n.ern.} = 0$ and $f_{PE,ern.} = 1.$

    *Nuclear, Unknown and Battery_Discharge* were assumed with the *Electricity Mix* of $f_{PE,n.ern.} = 0.28$ and $f_{PE,ern.} = 1.32$
    """
    for src in sources:
        df[src] = df[src] * PE_factors_OIB2019[src][type]
    df["total_consumption_avg"] = df.sum(axis=1)
    return df

def pe_factors(df = fetch_common()):
    """returns"""
    df.rename(columns={a: b for a, b in zip(pcs, sources)}, inplace=True)
    pc = "total_consumption_avg"
    PE = primary_energy(df[sources], "fPE")
    fPE = PE[pc] / df[pc]
    # %%
    PEnern = primary_energy(df[sources], "fPE,n.ern.")
    fPEnern = PEnern[pc] / df[pc]
    # %%
    PEern = primary_energy(df[sources], "fPE,ern.")
    fPEern = PEern[pc] / df[pc]
    return pd.DataFrame({"fPE":fPE,
                         "fPE,n.ern.":fPEnern,
                         "fPE,ern.":fPEern}
                        )

