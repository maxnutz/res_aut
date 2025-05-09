# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Solves linear optimal power flow for a network iteratively while updating reactances.

Relevant Settings
-----------------

.. code:: yaml

    solving:
        tmpdir:
        options:
            formulation:
            clip_p_max_pu:
            load_shedding:
            noisy_costs:
            nhours:
            min_iterations:
            max_iterations:
            skip_iterations:
            track_iterations:
        solver:
            name:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity_cf`, :ref:`solving_cf`, :ref:`plotting_cf`

Inputs
------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`prepare`

Outputs
-------

- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Solved PyPSA network including optimisation results

    .. image:: ../img/results.png
        :scale: 40 %

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.
The optimization is based on the ``pyomo=False`` setting in the :func:`network.lopf` and  :func:`pypsa.linopf.ilopf` function.
Additionally, some extra constraints specified in :mod:`prepare_network` are added.

Solving the network in multiple iterations is motivated through the dependence of transmission line capacities and impedances.
As lines are expanded their electrical parameters change, which renders the optimisation bilinear even if the power flow
equations are linearized.
To retain the computational advantage of continuous linear programming, a sequential linear programming technique
is used, where in between iterations the line impedances are updated.
Details (and errors made through this heuristic) are discussed in the paper

- Fabian Neumann and Tom Brown. `Heuristics for Transmission Expansion Planning in Low-Carbon Energy System Models <https://arxiv.org/abs/1907.10548>`_), *16th International Conference on the European Energy Market*, 2019. `arXiv:1907.10548 <https://arxiv.org/abs/1907.10548>`_.

.. warning::
    Capital costs of existing network components are not included in the objective function,
    since for the optimisation problem they are just a constant term (no influence on optimal result).

    Therefore, these capital costs are not included in ``network.objective``!

    If you want to calculate the full total annual system costs add these to the objective value.

.. tip::
    The rule :mod:`solve_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`solve_network`.

"""

import logging
from _helpers import configure_logging

import numpy as np
import pandas as pd
import re

import pypsa
from pypsa.linopf import (get_var, define_constraints, linexpr, join_exprs,
                          network_lopf, ilopf)

from pathlib import Path
from vresutils.benchmark import memory_logger

logger = logging.getLogger(__name__)


def prepare_network(n, solve_opts):

    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

    if solve_opts.get('load_shedding'):
        n.add("Carrier", "Load")
        buses_i = n.buses.query("carrier == 'AC'").index
        n.madd("Generator", buses_i, " load",
               bus=buses_i,
               carrier='load',
               sign=1e-3, # Adjust sign to measure p and p_nom in kW instead of MW
               marginal_cost=1e2, # Eur/kWh
               # intersect between macroeconomic and surveybased
               # willingness to pay
               # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
               p_nom=1e9 # kW
               )

    if solve_opts.get('noisy_costs'):
        for t in n.iterate_components(n.one_port_components):
            #if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if 'marginal_cost' in t.df:
                t.df['marginal_cost'] += (1e-2 + 2e-3 *
                                          (np.random.random(len(t.df)) - 0.5))

        for t in n.iterate_components(['Line', 'Link']):
            t.df['capital_cost'] += (1e-1 +
                2e-2*(np.random.random(len(t.df)) - 0.5)) * t.df['length']

    if solve_opts.get('nhours'):
        nhours = solve_opts['nhours']
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760. / nhours
    
    #print('changing cap of ror in AUT')
    #logger.warning('Change cap of ror in AUT to 5000')
    #n.generators.at['AT0 0 ror', 'p_nom'] = 4.523e+04 # 1.5e+8/4.e+3
    #print('cap of ror in AUT: p_nom = ', n.generators.p_nom.filter(like = 'AT')) #if not changed try
    return n
#####################################################
def adapt_cap(n, config):
    ror_cap = config['changed_installed_cap'].get('ror')
    solar_cap = config['changed_installed_cap'].get('max_solar')
    if not ror_cap is None:
        logger.warning('Change cap of ror in AUT to ', ror_cap)
        n.generators.at['AT0 0 ror', 'p_nom'] = ror_cap
    if not solar_cap is None:
        logger.warning('Change maximum installable capacity of solar in AUT to', solar_cap)
        n.generators.at['AT0 0 solar', 'p_nom_max'] = solar_cap
    print(n.generators.p_nom.filter(like = 'AT'), '\n', n.generators.p_nom_max.filter(like = 'AT'))
    return n

def add_line_constraints(n,config): # restricts s_nom_max to 2 * s_nom for lines from/to austria
    a = config['line_constraint'].get('factor')
    indices = (n.lines.loc[n.lines.bus0=='AT0 0'].index).append(n.lines.loc[n.lines.bus1=='AT0 0'].index)
    for i in indices:
    	n.lines.at[i, 's_nom_max'] = n.lines.at[i, 's_nom']*a
    logger.warning('restrict maximum s_nom from/to AT to x times of installed s_nom',a)
#######################################################
def add_CCL_constraints(n, config):
    agg_p_nom_limits = config['electricity'].get('agg_p_nom_limits')

    try:
        agg_p_nom_minmax = pd.read_csv(agg_p_nom_limits,
                                       index_col=list(range(2)))
    except IOError:
        logger.exception("Need to specify the path to a .csv file containing "
                          "aggregate capacity limits per country in "
                          "config['electricity']['agg_p_nom_limit'].")
    logger.info("Adding per carrier generation capacity constraints for "
                "individual countries")

    gen_country = n.generators.bus.map(n.buses.country)
    # cc means country and carrier
    p_nom_per_cc = (pd.DataFrame(
                    {'p_nom': linexpr((1, get_var(n, 'Generator', 'p_nom'))),
                    'country': gen_country, 'carrier': n.generators.carrier})
                    .dropna(subset=['p_nom'])
                    .groupby(['country', 'carrier']).p_nom
                    .apply(join_exprs))
    minimum = agg_p_nom_minmax['min'].dropna()
    if not minimum.empty:
        minconstraint = define_constraints(n, p_nom_per_cc[minimum.index],
                                           '>=', minimum, 'agg_p_nom', 'min')
    maximum = agg_p_nom_minmax['max'].dropna()
    if not maximum.empty:
        maxconstraint = define_constraints(n, p_nom_per_cc[maximum.index],
                                           '<=', maximum, 'agg_p_nom', 'max')


def add_EQ_constraints(n, o, scaling=1e-1):
    float_regex = "[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == 'c':
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = n.snapshot_weightings.generators @ \
           n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    inflow = n.snapshot_weightings.stores @ \
             n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    inflow = inflow.reindex(load.index).fillna(0.)
    rhs = scaling * ( level * load - inflow )
    lhs_gen = linexpr((n.snapshot_weightings.generators * scaling,
                       get_var(n, "Generator", "p").T)
              ).T.groupby(ggrouper, axis=1).apply(join_exprs)
    lhs_spill = linexpr((-n.snapshot_weightings.stores * scaling,
                         get_var(n, "StorageUnit", "spill").T)
                ).T.groupby(sgrouper, axis=1).apply(join_exprs)
    lhs_spill = lhs_spill.reindex(lhs_gen.index).fillna("")
    lhs = lhs_gen + lhs_spill
    define_constraints(n, lhs, ">=", rhs, "equity", "min")


def add_BAU_constraints(n, config):
    mincaps = pd.Series(config['electricity']['BAU_mincapacities'])
    lhs = (linexpr((1, get_var(n, 'Generator', 'p_nom')))
           .groupby(n.generators.carrier).apply(join_exprs))
    define_constraints(n, lhs, '>=', mincaps[lhs.index], 'Carrier', 'bau_mincaps')


def add_SAFE_constraints(n, config):
    peakdemand = (1. + config['electricity']['SAFE_reservemargin']) *\
                  n.loads_t.p_set.sum(axis=1).max()
    conv_techs = config['plotting']['conv_techs']
    exist_conv_caps = n.generators.query('~p_nom_extendable & carrier in @conv_techs')\
                       .p_nom.sum()
    ext_gens_i = n.generators.query('carrier in @conv_techs & p_nom_extendable').index
    lhs = linexpr((1, get_var(n, 'Generator', 'p_nom')[ext_gens_i])).sum()
    rhs = peakdemand - exist_conv_caps
    define_constraints(n, lhs, '>=', rhs, 'Safe', 'mintotalcap')


def add_battery_constraints(n):
    nodes = n.buses.index[n.buses.carrier == "battery"]
    if nodes.empty or ('Link', 'p_nom') not in n.variables.index:
        return
    link_p_nom = get_var(n, "Link", "p_nom")
    lhs = linexpr((1,link_p_nom[nodes + " charger"]),
                  (-n.links.loc[nodes + " discharger", "efficiency"].values,
                   link_p_nom[nodes + " discharger"].values))
    define_constraints(n, lhs, "=", 0, 'Link', 'charger_ratio')
###AUSTRIAN_CONSTRAINTS###########################################################################
def constraint_wind(n, config):
    #print('#############\n\nStart constraint\n')
    config = n.config
    gen_constraints = config['minimal_generation']
    solar_var = get_var(n, 'Generator', 'p').filter(like='onwind').filter(like='AT')
    lhs = linexpr((1, solar_var)).sum(axis=1).sum(axis=0)
    rhs = (config['minimal_generation'].get('wind'))
    define_constraints(n, lhs, '>=', rhs, 'wind_constraint')
    print('WIND_constraints\n')

def constraint_windMAX(n, config):
    config = n.config
    gen_constraints = config['minimal_generation']
    solar_var = get_var(n, 'Generator', 'p').filter(like='onwind').filter(like='AT')
    lhs = linexpr((1, solar_var)).sum(axis=1).sum(axis=0)
    rhs = (config['minimal_generation'].get('wind'))*1.1
    define_constraints(n, lhs, '<=', rhs, 'wind_constraint')
    print('WIND_constraint_maximum\n')
        
def constraint_solar(n, config):
    config = n.config
    gen_constraints = config['minimal_generation']
    solar_var = get_var(n, 'Generator', 'p').filter(like='solar').filter(like='AT')
    lhs = linexpr((1, solar_var)).sum(axis=1).sum(axis=0)
    rhs = (config['minimal_generation'].get('solar'))
    define_constraints(n, lhs, '>=', rhs, 'solar_constraint')
    print('SOLAR_constraints\n')

def constraint_solarMAX(n, config):
    # in einem 10%-RANGE
    config = n.config
    gen_constraints = config['minimal_generation']
    #print(gen_constraints)
    solar_var = get_var(n, 'Generator', 'p').filter(like='solar').filter(like='AT')
    lhs = linexpr((1, solar_var)).sum(axis=1).sum(axis=0)
    rhs = (((config['minimal_generation'].get('solar'))+(config['minimal_generation'].get('solar'))*0.1))
    define_constraints(n, lhs, '<=', rhs, 'solar_constraint')
    print('SOLAR_constraint_maximum\n')

def constraint_ror(n, config):
    config = n.config
    gen_constraints = config['minimal_generation']
    ror_var = get_var(n, 'Generator', 'p').filter(like='ror').filter(like='AT') #+ get_var(n, 'Generator', 'p').filter(like='hydro').filter(like='AT')
    lhs = linexpr((1, ror_var)).sum(axis=1).sum(axis=0)
    rhs = (config['minimal_generation'].get('ror'))
    define_constraints(n, lhs, '>=', rhs, 'ror_constraint')
    print('ROR_constraints\n')

def constraint_rorMAX(n, config):
    config = n.config
    gen_constraints = config['minimal_generation']
    ror_var = get_var(n, 'Generator', 'p').filter(like='ror').filter(like='AT') #+ get_var(n, 'Generator', 'p').filter(like='hydro').filter(like='AT')
    lhs = linexpr((1, ror_var)).sum(axis=1).sum(axis=0)
    rhs = ((config['minimal_generation'].get('ror'))*1.02)
    define_constraints(n, lhs, '<=', rhs, 'ror_constraint')
    print('ROR_constraint_maximum\n')

def constraint_coal(n):
    coal_var = get_var(n, 'Generator', 'p').filter(like='coal').filter(like='AT')
    lhs = linexpr((1, coal_var)).sum(axis=1).sum(axis=0)
    rhs = 10
    define_constraints(n, lhs, '<=', rhs, 'coal_constraint')
    print('coal_constraint')



## GERMANY   
def DE_nuclear(n):
    nuclear = get_var(n, 'Generator', 'p').filter(like='nuclear').filter(like='DE')
    lhs = linexpr((1, nuclear)).sum(axis=1).sum(axis=0)
    rhs = 10
    define_constraints(n, lhs, '<=', rhs, 'DE_nuclear_constraint')
    print('DE_nuclear_constraints')

def DE_solar(n, config):
    solar = get_var(n, 'Generator', 'p').filter(like='solar').filter(like='DE')
    lhs = linexpr((1, solar)).sum(axis=1).sum(axis=0)
    rhs = 9.e7
    define_constraints(n, lhs, '>=', rhs, 'DE_solar_constraint')
def DE_onwind(n, config):
    solar = get_var(n, 'Generator', 'p').filter(like='onwind').filter(like='DE')
    lhs = linexpr((1, solar)).sum(axis=1).sum(axis=0)
    rhs = 1.4e8
    define_constraints(n, lhs, '>=', rhs, 'DE_onwind_constraint')
def DE_offwind(n, config):
    solar = get_var(n, 'Generator', 'p').filter(like='offwind').filter(like='DE')
    lhs = linexpr((1, solar)).sum(axis=1).sum(axis=0)
    rhs = 7.9e7
    define_constraints(n, lhs, '>=', rhs, 'DE_offwind_constraint')
    print('DE_VRE_constraints')

## SCHWEIZ
def CH_nuclear(n, config):
    nuclear = get_var(n, 'Generator', 'p').filter(like='nuclear').filter(like='CH')
    lhs = linexpr((1, nuclear)).sum(axis=1).sum(axis=0)
    rhs = 5.3e6 #1/3 
    define_constraints(n, lhs, '<=', rhs, 'CH_nuclear_constraint')
    print('CH_nuclear_constraints')
## ITALY
def IT_renewables(n, config):
    solar_var = get_var(n, 'Generator', 'p').filter(like = 'solar').filter(like='IT')
    wind_var = get_var(n, 'Generator', 'p').filter(like = 'onwind').filter(like='IT')
    lhs = linexpr((1, solar_var)).sum(axis=1).sum(axis=0) + linexpr((1, wind_var)).sum(axis=1).sum(axis=0)
    rhs = 2.5e8
    define_constraints(n, lhs, '>=', rhs, 'IT_RES-E_constraint')
    print('IT_RES-E_constraint')
## SLOVAKIA
def SK_solar(n, config):
    solar_var = get_var(n, 'Generator', 'p').filter(like = 'solar').filter(like = 'SK')
    lhs = linexpr((1, solar_var)).sum(axis=1).sum(axis=0)
    rhs = 1.2e6
    define_constraints(n, lhs, '>=', rhs, 'SK_solar_constraint')
    print('SK_solar_constraint')

def SK_wind(n, config):
    wind_var = get_var(n, 'Generator', 'p').filter(like = 'onwind').filter(like = 'SK')
    lhs = linexpr((1, wind_var)).sum(axis=1).sum(axis=0)
    rhs = 1e6
    define_constraints(n, lhs, '>=', rhs, 'SK_wind_constraint')
    print('SK_Wind_constraint')
## HUNGARY
def HU_nuclear(n, config):
    nuclear_var = get_var(n, 'Generator', 'p').filter(like= 'nuclear').filter(like='HU')
    lhs = linexpr((1, nuclear_var)).sum(axis=1).sum(axis=0)
    rhs = 3e7
    define_constraints(n, lhs, '>=', rhs, 'HU_nuclear_constraint')
    print('HU_nuclear_constraint')
    
def HU_gas(n, config):
    CCGT_var = get_var(n, 'Generator', 'p').filter(like='CCGT').filter(like='HU')
    PCGT_var = get_var(n, 'Generator', 'p').filter(like='PCGT').filter(like='HU')
    lhs = linexpr((1, CCGT_var)).sum(axis=1).sum(axis=0) + linexpr((1, PCGT_var)).sum(axis=1).sum(axis=0)
    rhs = 2.5e7
    define_constraints(n, lhs, '>=', rhs, 'HU_gas_constraint')
    print('HU_gas_constraint')

def HU_rens(n, config):
    solar_var = get_var(n, 'Generator', 'p').filter(like='solar').filter(like='HU')
    wind_var = get_var(n, 'Generator', 'p').filter(like='onwind').filter(like='HU')
    hydro_var = get_var(n, 'Generator', 'p').filter(like='ror').filter(like='HU') + get_var(n, 'Generator', 'p').filter(like='hydro').filter(like='HU')
    lhs = linexpr((1, solar_var)).sum(axis=1).sum(axis=0) + linexpr((1, wind_var)).sum(axis=1).sum(axis=0) + linexpr((1, hydro_var)).sum(axis=1).sum(axis=0)
    rhs = 9.17e6
    define_constraints(n, lhs, '<=', rhs, 'HU_renewable_constraints')
    print('HU_renewable_constraints')
## CZECH REPUBLIC
def CZ_nuclear(n, config):
    nuclear_var = get_var(n, 'Generator', 'p').filter(like='solar').filter(like='CZ')
    lhs = linexpr((1, nuclear_var)).sum(axis=1).sum(axis=0)
    rhs = 4.62e7
    define_constraints(n, lhs, '<=', rhs, 'CZ_nuclear_constraint')
    print('CZ_nuclear_constraint')

##############################################################################################

def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to ``pypsa.linopf.network_lopf``.
    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if 'BAU' in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if 'SAFE' in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if 'CCL' in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)
    add_battery_constraints(n)
    add_line_constraints(n, config)
    # supplementary constraints for production scenarios
    constraint_solar(n, config)
    constraint_solarMAX(n, config)
    constraint_wind(n, config)
    constraint_windMAX(n, config)
    constraint_ror(n, config)
    constraint_rorMAX(n, config)
    constraint_coal(n)
    DE_nuclear(n)
    #DE_solar(n, config)
    #DE_onwind(n, config)
    DE_offwind(n,config)
    CH_nuclear(n, config)
    IT_renewables(n, config)
    SK_solar(n, config)
    SK_wind(n, config)
    #HU_nuclear(n, config)
    #HU_gas(n, config)
    HU_rens(n, config)
    CZ_nuclear(n, config)


def solve_network(n, config, opts='', **kwargs):
    solver_options = config['solving']['solver'].copy()
    solver_name = solver_options.pop('name')
    cf_solving = config['solving']['options']
    track_iterations = cf_solving.get('track_iterations', False)
    min_iterations = cf_solving.get('min_iterations', 4)
    max_iterations = cf_solving.get('max_iterations', 6)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if cf_solving.get('skip_iterations', False):
        network_lopf(n, solver_name=solver_name, solver_options=solver_options,
                     extra_functionality=extra_functionality, **kwargs)
    else:
        ilopf(n, solver_name=solver_name, solver_options=solver_options,
              track_iterations=track_iterations,
              min_iterations=min_iterations,
              max_iterations=max_iterations,
              extra_functionality=extra_functionality, **kwargs)
    return n


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('solve_network', network='elec', simpl='',
                                  clusters='5', ll='copt', opts='Co2L-BAU-CCL-24H')
    configure_logging(snakemake)

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    opts = snakemake.wildcards.opts.split('-')
    solve_opts = snakemake.config['solving']['options']

    fn = getattr(snakemake.log, 'memory', None)
    with memory_logger(filename=fn, interval=30.) as mem:
        n = pypsa.Network(snakemake.input[0])
        n = prepare_network(n, solve_opts)
        n = adapt_cap(n, config=snakemake.config) ################
        n = solve_network(n, config=snakemake.config, opts=opts,
                          solver_dir=tmpdir,
                          solver_logfile=snakemake.log.solver)
        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
