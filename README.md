# Mechanism Reduction with GA and GNN

This repository implements an end-to-end pipeline for reducing chemical
reaction mechanisms.  A genetic algorithm (GA) searches for a minimal set of
species while a graph neural network (GNN) guides the initial population based
on species importance.  Reactor simulations are performed with Cantera and the
quality of a reduction is assessed using progress variables and ignition delay
metrics.

## Running the Pipeline

```
python -m testing.run_tests --out results --steps 200 --tf 1.0
```

The command loads the `data/gri30.yaml` mechanism, executes a batch reactor
simulation and applies the GA reduction.  CSV files and plots are written to
`results/`.  Additional debugging information is stored in
`debug_fitness.csv` and generation-wise PV/temperature plots.

## Dependencies

- [Cantera](https://cantera.org)
- [NetworkX](https://networkx.org)
- [torch_geometric](https://pytorch-geometric.readthedocs.io)
- [Matplotlib](https://matplotlib.org)

Install them with pip if necessary:

```
pip install cantera networkx torch torch_geometric matplotlib
```

## Repository Layout

```
data/               Example mechanism files and weights
mechanism/          Mechanism parsing and editing utilities
reactor/            Reactor models using Cantera
progress_variable.py  Progress variable computation
metrics.py          Error metrics and ignition delay
metaheuristics/     Genetic algorithm implementation
gnn/                Simple GNN models for species scoring
testing/            CLI entry points and pipeline utilities
visualizations/     Plotting helpers
```
