# === testing/run_tests.py ===

import argparse
import os
import cantera as ct
from testing.pipeline import full_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", default="data/gri30.yaml", help="Path to mechanism YAML file")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--steps", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--tf", type=float, default=0.5, help="Final simulation time [s]")
    parser.add_argument("--phi", type=float, default=1.0, help="Equivalence ratio")
    parser.add_argument(
        "--preset",
        default="methane_air",
        choices=["methane_air", "HR1", "HR2", "HR3"],
        help="Mixture preset",
    )
    parser.add_argument("--T0", type=float, default=1500.0, help="Initial temperature [K]")
    parser.add_argument("--p0", type=float, default=float(ct.one_atm), help="Initial pressure [Pa]")
    parser.add_argument("--log-times", action="store_true", help="Use log-spaced time sampling")
    parser.add_argument("--isothermal", action="store_true", help="Use isothermal reactor for HR presets")
    parser.add_argument("--mode", choices=["0d", "1d"], default="0d", help="Reactor mode")
    parser.add_argument("--diluent", default="N2", help="Diluent species for methane/air mixing")
    parser.add_argument("--L", type=float, default=0.5, help="Reactor length [m]")
    parser.add_argument("--D", type=float, default=0.05, help="Reactor diameter [m]")
    parser.add_argument("--mdot", type=float, default=0.1, help="Mass flow [kg/s]")
    parser.add_argument("--U", type=float, default=0.0, help="Overall heat transfer coefficient [W/m^2/K]")
    parser.add_argument("--Tw", type=float, default=None, help="Wall temperature [K] for heat loss")
    parser.add_argument("--plasma-length", type=float, default=0.0, help="Length of plasma zone [m]")
    parser.add_argument("--T-plasma-out", type=float, default=None, help="Target temperature after plasma zone [K]")
    parser.add_argument("--radical-seed", default=None, help="Comma-separated radicals (e.g. H:0.01,OH:0.005)")
    parser.add_argument("--envelopes", default="envelopes.json", help="Envelope specification file")
    parser.add_argument("--train-cases", default="auto", help="Training case specification or 'auto'")
    parser.add_argument("--train-weights", default=None, help="Comma list or named weights for training cases")
    parser.add_argument("--topn-species", type=int, default=6, help="Top-N species for axial overlays")

    # === NEW ARGUMENTS for label building and aggressiveness ===
    parser.add_argument("--alpha", type=float, default=0.8, help="Weight between LOO and flux importance")
    parser.add_argument("--tf-short", type=float, default=None, help="Shortened tf for LOO runs")
    parser.add_argument("--steps-short", type=int, default=None, help="Steps for LOO runs")
    parser.add_argument("--no-cache-labels", action="store_true", help="Recompute labels instead of loading cache")
    parser.add_argument("--min-species", type=int, default=None, help="Minimum species allowed in GA")
    parser.add_argument("--max-species", type=int, default=None, help="Maximum species allowed in GA")
    parser.add_argument("--target-species", type=int, default=None, help="Target species count")
    parser.add_argument("--generations", type=int, default=60, help="GA generations")
    parser.add_argument("--population", type=int, default=40, help="GA population size")
    parser.add_argument("--mutation", type=float, default=0.25, help="GA mutation rate")
    parser.add_argument(
        "--focus",
        default="auto",
        choices=["none", "auto", "window"],
        help="Plot focus window setting",
    )
    parser.add_argument(
        "--focus-window",
        nargs=2,
        type=float,
        default=None,
        help="Custom focus window tmin tmax when --focus=window",
    )

    # Compatibility placeholders for downstream tests
    parser.add_argument("--fitness-mode", default="standard")
    parser.add_argument("--tol-pv", type=float, default=0.05)
    parser.add_argument("--tol-delay", type=float, default=0.05)
    parser.add_argument("--tol-timescale", type=float, default=0.05)
    parser.add_argument("--tol-resid", type=float, default=0.05)
    parser.add_argument("--phaseB-unlock", action="store_true")
    parser.add_argument("--report-grid", default=None)

    args = parser.parse_args()

    seed_dict = None
    if args.radical_seed:
        seed_dict = {}
        for token in args.radical_seed.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                name, value = token.split(":")
                seed_dict[name.strip()] = float(value)
            except ValueError:
                raise ValueError(f"Invalid radical seed entry '{token}'. Use species:value format.")

    full_pipeline(
        args.mechanism,
        args.out,
        steps=args.steps,
        tf=args.tf,
        phi=args.phi,
        preset=args.preset,
        T0=args.T0,
        p0=args.p0,
        log_times=args.log_times,
        alpha=args.alpha,
        tf_short=args.tf_short,
        steps_short=args.steps_short,
        cache_labels=not args.no_cache_labels,
        isothermal=args.isothermal,
        min_species=args.min_species,
        max_species=args.max_species,
        target_species=args.target_species,
        generations=args.generations,
        population=args.population,
        mutation=args.mutation,
        focus=args.focus,
        focus_window=tuple(args.focus_window) if args.focus_window else None,
        report_grid=args.report_grid,
        mode=args.mode,
        diluent=args.diluent,
        L=args.L,
        D=args.D,
        mdot=args.mdot,
        U=args.U,
        Tw=args.Tw,
        fitness_mode=args.fitness_mode,
        tol_pv=args.tol_pv,
        tol_delay=args.tol_delay,
        tol_timescale=args.tol_timescale,
        tol_resid=args.tol_resid,
        plasma_length=args.plasma_length,
        T_plasma_out=args.T_plasma_out,
        radical_seed=seed_dict,
        envelopes_path=args.envelopes,
        train_cases=args.train_cases,
        train_weights=args.train_weights,
        topn_species=args.topn_species,
    )
    print(f"\n✅ Pipeline complete. Results written to '{args.out}'\n")


if __name__ == "__main__":
    main()
