# res_aut repo for Master-thesis

## Setup
- Set up conda-environment (in folder with `environment.yml`)
~~~
conda env create
~~~
- Install Solver (eg. Gurobi-Solver with free academic licence)
- retrieve databundle (in folder `pypsa-eur`)
~~~
cp config.default.yaml config.yaml
snakemake -j 1 results/networks/elec_s_11_ec_lcopt_Co2L-1H.nc
~~~
after this run, one can change `retrieve databundle` to false in `config.yml`.
Environment for energy network calculations in Austria. The Workflow strongly depends on the pypsa-eur Workflow. Descriptions and alle Workflow-Informations can be found in the jupyter-notebook in "Dokument/Workflow". 

This repository exists for intern use, probably the code won't run without additional explanation. There will be a new repository with the final workflow within the next weeks and it will be linked in here.

Changed pypsa-eur Scripts:
- cluter network.py
- prepare_network.py
- solve_network.py
- _helpers.py
