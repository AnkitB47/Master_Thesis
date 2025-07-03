# Master Thesis: Mechanism Reduction Toolkit

This project provides a Python implementation for reducing chemical reaction mechanisms using graph algorithms, metaheuristics, and neural networks.  It reproduces much of the functionality of the original MATLAB framework found in the `MECH` folder and integrates portions of Arnab's GA project.

## Project Layout

```
/data               Example mechanism files
/graphs             Exported graphs in GraphML
/gnn                Graph neural network models
/mechanism          Mechanism parsing and editing utilities
/metaheuristics     GA, ABC, Bees and Branch-and-Bound algorithms
/reactor            Reactor models using Cantera
/testing            Command line testing scripts
/visualizations     Plotting utilities
/results            Output data and figures
```

## Example Usage

Run the full testing pipeline:

```bash
python -m testing.run_tests --mechanism data/gri30.yaml --out results
```

This command executes all metaheuristics, compares reduced mechanisms to the full one and stores metrics and plots in the `results` folder.

Generated figures include convergence curves for each metaheuristic and
comparison bar charts. Example plots are shown below:

![Convergence](results/convergence.png)
![Comparison](results/comparison.png)

## Installation

Required packages are listed below.  Install with `pip`:

```
pip install cantera numpy scipy pandas matplotlib networkx torch torch_geometric
```

## Modules

- **mechanism** – parse CTI/CK mechanisms using Cantera and remove species/reactions.
- **graph** – construct species–reaction graphs and export in GraphML format.
- **reactor** – batch, piston and network reactor solvers and 1-D flames.
- **progress_variable** – compute progress variables from reactor results.
- **metrics** – distance metrics for comparing reactor runs.
- **timescales** – basic timescale analysis.
- **metaheuristics** – GA, ABC, Bees and Branch-and-Bound algorithms for mechanism reduction.
- **gnn** – graph neural network models (GCN and GAT) for scoring species.
- **testing** – scripts to run example simulations and measure reduction errors.

## Thesis Pipeline
1. Load a mechanism using `mechanism.Mechanism`.
2. Build the species graph with `graph.construction.build_species_graph`.
3. Run reactor simulations with modules from `reactor`.
4. Compute progress variables and errors.
5. Optimise species selection via GA, ABC, Bees or BnB metaheuristics.
6. Optionally train a GNN model to predict species importance.
7. Compute timescales and distance metrics for evaluation.

All results and plots should be written into the `results` and `visualizations` folders.
