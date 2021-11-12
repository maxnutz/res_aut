from FLUCCOplus.utils import *
from FLUCCOplus.notebooks import *

from matplotlib.figure import Figure

from dataclasses import dataclass, field, fields
@dataclass()
class Variable:
    name:str
    ylabel:str
    agg:object
    ylabel_agg:str



@log
def average(df, vars):
    fig = Figure(figsize=(15, 15))
    ax = fig.subplots(len(vars), 3)
    for col, var in enumerate(vars):
        sns.lineplot(x=df.index.hour, y=var.name, data=df, hue=df.index.year, ax=ax[col][0])
        sns.lineplot(x=df.index.month, y=var.name, data=df, hue=df.index.year, ax=ax[col][1])
        # df.groupby(df.index.year)[var.name].agg(var.agg).plot(kind="bar", rot=0, ax=ax[col][2])
        sns.barplot(x=df.index.year.unique(), y=var.name, data=df.groupby(df.index.year).agg(var.agg), ax=ax[col][2])
        ax[col][0].set(xlabel="Durchschnittliche Stunde", ylabel=var.ylabel)
        ax[col][1].set(xlabel="Monatsmittel", ylabel=var.ylabel)
        ax[col][2].set(xlabel="Jahr", ylabel=var.ylabel_agg)
    return fig

def average_sources(df, var):

    fig = Figure(figsize=(15, 15))
    ax = fig.subplots(1, 3)
    df.groupby(df.index.month)[var.name].agg(var.agg).plot(kind="bar",stacked=True, ax=ax[2])
    # df.groupby(df.index.year)[var.name].agg(var.agg).plot(kind="bar", rot=0, ax=ax[col][2])
    sns.barplot(x=df.index.year.unique(), y=var.name, data=df.groupby(df.index.year).agg(var.agg), ax=ax[2])
    ax[0].set(xlabel="Durchschnittliche Stunde", ylabel=var.ylabel)
    ax[1].set(xlabel="Monatsmittel", ylabel=var.ylabel)
    ax[2].set(xlabel="Jahr", ylabel=var.ylabel_agg)



@logg
def plot_41_ec_eb(df, carriers, uses, year):

    # %%
    return df

def emissionyear(rs, oib_co2, fig, ax, var="carbon_intensity_avg",year=2015):
    # em co2 corriduor
    sns.lineplot(x=rs.index.week, y=var, data=rs, color="black", ax=ax, ci=99.9)
    # em yearly
    co2_mean_em = rs.loc[rs.index.year==year, var].mean()
    pd.Series([co2_mean_em for m in rs.index.month], rs.index.isocalendar().week).plot(ax=ax, color="black", linewidth = 3)
    # em all years average
    pd.Series([234.42 for m in rs.index.month], rs.index.isocalendar().week).plot(ax=ax, color="black",
                                                                                       linewidth=2)
    # oib 18 monthly
    oib18 = pd.Series([oib_co2.loc[m-1] for m in rs.index.month], rs.index.isocalendar().week)
    oib18[3:-3].plot(color="red", linewidth=2, ax=ax) #indexin weirdness (3:-3)
    # oib 19 yearly
    oib19 = pd.Series([227 for m in rs.index.month], rs.index.isocalendar().week)
    oib19.plot(color="darkred", linewidth=3, ax=ax)


def plot_OIBCO2_comparison(rs, oib, years=[2015,2016,2017,2018,2019]):
    fig, ax = plt.subplots(1, len(years), figsize=(4*len(years),5), sharey=True)
    for i, y in enumerate(years):
        emissionyear(rs.loc[rs.index.year == y], oib_co2=oib, fig=fig, ax=ax[i], year=y)
        ax[i].set_xticklabels([])
        ax[i].set_xticks(np.linspace(0,54,7))
        ax[i].set_xlim(0,54)
        ax[i].set_ylim(50,400)
        ax[i].set_xlabel(str(y), size=12)

    ax[0].set_ylabel('CO$_2$-Intensität [g/kWh$_e$$_l$]')
    ax[0].legend(["Measurement data (EM)", "annual average (EM)", "2015-2018 average (EM)","OIB Rl6 Monthly (2018)","OIB RL6 2019"], loc='lower left', fontsize=12)
    fig.tight_layout()
    return fig

def plot_HDW(df,
             var="carbon_intensity_avg",
             ylabel="Carbon emissions [g$_{CO2eq}$/kWh]",
             xlabel="",
             colors=["darkgrey", "black", "black"],
             legend=["Hourly","Daily average", "Weekly average"],
             xlim=("2019-01-01", "2019-12-31"),
             figsize=(10,8),
             fig=None,
             ax=None):
    """
    Plots the hourly, daily average and weekly average of a given df variable
    """

    df_daily = df.resample("D").mean()
    df_weekly = df.resample("W").mean()
    # df_monthly = df.resample("M").mean()
    if fig == None or ax == None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    df[var].plot(ax=ax, color=colors[0], marker='.', alpha=0.3, linestyle='None', legend=False)
    ax.set_xlabel(xlabel)

    df_daily[var].plot(ax=ax, color=colors[1], alpha=0.8, legend=False)
    ax.set_xlabel(xlabel)

    df_weekly[var].plot(color=colors[2], linewidth=1.5, ax=ax, legend=False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_xlim(xlim[0], xlim[1])
    ax.legend(legend, loc='lower left', fontsize=12)
    fig.tight_layout()
    return fig