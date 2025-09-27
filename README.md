# 🔬 Mechanism Reduction using Genetic Algorithms and Graph Neural Networks (GA-GNN)

This repository implements a complete pipeline for **chemical reaction mechanism reduction** using a combination of:

- 🧬 **Genetic Algorithms (GA)** for selecting optimal species subsets,
- 🧠 **Graph Neural Networks (GNN)** for learning species importance and guiding GA initialization,
- 🔥 **Cantera** reactor simulations to validate physical fidelity of reduced models.

Our fully functional **Prototype v1** performs end-to-end reduction, evaluates the quality using ignition delay and progress variable (PV) error, and exports CSVs and plots for detailed analysis.

---

## ♨️ HP-POX feed compatibility

The HP-POX benchmark feeds include trace quantities of heavier hydrocarbons such as the butane isomers (`IC4H10`, `NC4H10`). Because widely used reference mechanisms like **GRI-30** omit these species, the 1-D plug-flow solver automatically reconciles feeds with the selected mechanism. Missing C4+ components are lumped into propane (`C3H8`) by default—falling back to methane if propane is unavailable—so that carbon accounting remains defensible when switching between mechanisms.

---

## 🚀 How to Run the Pipeline

```bash
python -m testing.run_tests --mechanism data/gri30.yaml --out results --steps 200 --tf 5.0
````

* `--mechanism`: Path to the full chemical mechanism file (default: `data/gri30.yaml`)
* `--out`: Folder where outputs are saved (auto-created if missing)
* `--steps`: Integration steps for the batch reactor simulation
* `--tf`: Final time for simulation (in seconds)

---

## 📁 What Gets Saved in `results/`

* `ga_fitness.csv` — Best fitness score per generation
* `debug_fitness.csv` — Full fitness breakdown for all GA individuals (PV error, delay diff, penalties)
* `gnn_scores.csv` — Species importance scores predicted by GNN
* `convergence.png` — GA fitness evolution over generations
* `profiles.png` — Mass fraction comparison of key species (Full vs Reduced)
* `ignition_delay.png` — Ignition delay for reduced mechanism
* `pv_error.png` — PV error (progress variable deviation)

---

## 🧠 GNN Integration

The GNN model is trained on species graph constructed from Cantera with:

* Node features: out-degree, in-degree, and normalized thermo ranges
* Trained using known species importance from full simulation
* Predicts importance scores saved to `gnn_scores.csv`

These scores are used to seed GA with biologically/chemically meaningful individuals.

---

## ⚙️ Dependencies

Make sure to install the following Python libraries:

```bash
pip install cantera networkx torch torch_geometric matplotlib
```

Python ≥ 3.9 is recommended. Tested with `cantera==3.0.0`.

---

## 🗂️ Project Structure

```
data/                 # Sample mechanisms and species weights
gnn/                  # GCN/GAT models and GNN training utilities
graph/                # Graph construction from Cantera mechanism
mechanism/            # Species and reaction editing modules
metaheuristics/       # Genetic algorithm and operators
reactor/              # Cantera-based reactor simulators (batch, flame, etc.)
testing/              # CLI scripts and pipeline integration
visualizations/       # Plotting functions for output analysis
progress_variable.py  # Progress variable evaluation
metrics.py            # Error metrics: PV error, ignition delay
```

---

## 📈 Example Output Quality

With `tf=5.0`, `generations=25`, and `GNN epochs=15`:

* ✅ PV Error: \~0.0175
* ✅ Ignition Delay: \~0.025 s (realistic)
* ✅ Species Profiles: CH₄, O₂, CO₂ match full mechanism closely
* ✅ GNN Scores show strong separation between critical and inert species
* ✅ GA shows smooth fitness convergence across generations

---

## 🧪 Extensions You Can Try

| Goal                           | How                                    |
| ------------------------------ | -------------------------------------- |
| More aggressive reduction      | Increase GA generations, mutations     |
| Faster convergence             | Raise GNN training epochs              |
| Different reactor types        | Replace with `piston.py` or `flame.py` |
| Broader temperature validation | Use multiple `Y0` and `T` settings     |

