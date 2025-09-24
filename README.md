# 🔬 Mechanism Reduction using Genetic Algorithms and Graph Neural Networks (GA-GNN)

This repository implements a complete pipeline for **chemical reaction mechanism reduction** using a combination of:

- 🧬 **Genetic Algorithms (GA)** for selecting optimal species subsets,
- 🧠 **Graph Neural Networks (GNN)** for learning species importance and guiding GA initialization,
- 🔥 **Cantera** reactor simulations to validate physical fidelity of reduced models.

Our fully functional **Prototype v1** performs end-to-end reduction, evaluates the quality using ignition delay and progress variable (PV) error, and exports CSVs and plots for detailed analysis.

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

## 🧵 1D Plug-Flow Reactor mode (WP2/WP5/WP6)

`--mode 1d` activates a tubular plug-flow reactor (PFR) with optional heat loss and plasma surrogates. The pipeline reads
`envelopes.json` to sample POX/HP-POX/CO₂-recycle operating windows, runs GA–GNN reduction under threshold fitness, and
emits additional artefacts:

* `pfr_profiles.csv` – axial temperature/species overlays (full vs. reduced) for all sampled cases.
* `pfr_kpis.csv` – CH₄/CO₂ conversion, H₂/CO ratios, ignition length, and pass/fail metrics per case.
* `robustness_1d.csv` / `robustness_plasma.csv` – threshold-metric audits for POX and plasma envelopes.
* `results/latex/robustness_pox.tex`, `results/latex/robustness_plasma.tex`, `results/latex/kpi_summary.tex` – tables ready for
  direct inclusion in WP reports.
* `visualizations/axial_overlay` & `visualizations/kpi_bars` – axial profile overlays (with ignition markers) and KPI bar charts
  across the envelope sweep.

### Minimal 1D examples

```bash
# POX nominal (adiabatic, φ=0.7, 10 bar, 700 K)
python -m testing.run_tests \
  --mode 1d --mechanism data/gri30.yaml --out results/pox_nominal \
  --phi 0.7 --T0 700 --p0 1.0e6 --L 0.8 --D 0.05 --mdot 0.12 --U 0.0 \
  --steps 400 --fitness-mode threshold --tol-pv 0.05 --tol-delay 0.05 --tol-timescale 0.05 --tol-resid 0.05

# Plasma surrogate (torch heating to 1500 K with radical seeding)
python -m testing.run_tests \
  --mode 1d --mechanism data/gri30.yaml --out results/plasma_demo \
  --phi 1.0 --T0 400 --p0 2.0e5 --L 0.6 --D 0.04 --mdot 0.08 \
  --plasma-length 0.1 --T-plasma-out 1500 --radical-seed "H:0.005,OH:0.002" \
  --steps 400 --fitness-mode threshold --tol-pv 0.05 --tol-delay 0.05 --tol-timescale 0.05 --tol-resid 0.05
```

Both commands populate the additional PFR CSVs, plots, and LaTeX tables described above while keeping the GA–GNN reduction
workflow consistent with the 0D baseline.

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

