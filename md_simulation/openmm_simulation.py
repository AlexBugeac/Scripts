#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
from multiprocessing import Pool
from functools import partial

# Headless plotting before importing matplotlib
if "--no-plots" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mdtraj as md

# Modern OpenMM imports
from openmm import app
import openmm as mm
from openmm import unit


#################################################################
# ARGUMENT PARSER
#################################################################
def get_args():
    parser = argparse.ArgumentParser(description="Parallel OpenMM MD + analysis")

    # MD inputs
    parser.add_argument("--inputs", nargs="+",
        help="Pairs: prmtop rst7 prmtop rst7 ...")
    parser.add_argument("--auto", action="store_true",
        help="Automatically detect *_pipeline folders")

    # Analysis-only
    parser.add_argument("--analyze-only", action="store_true",
        help="Run analysis only (skip MD)")
    parser.add_argument("--traj", help="Trajectory DCD file")
    parser.add_argument("--prmtop", help="PRMTOP file")
    parser.add_argument("--output-prefix", help="Where to save plots")

    # MD parameters
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--temp", type=float, default=300.0)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--interval", type=int, default=2000)
    parser.add_argument("--parallel", type=int, default=3)

    parser.add_argument("--no-plots", action="store_true")

    return parser.parse_args()


#################################################################
# AUTO-DETECT INPUT PIPELINE DIRECTORIES
#################################################################
def autodetect_inputs():
    pairs = []
    for folder in sorted(os.listdir(".")):
        if folder.endswith("_pipeline") and os.path.isdir(folder):
            prmtop = None
            rst7 = None
            for f in os.listdir(folder):
                if f.endswith(".prmtop"):
                    prmtop = os.path.join(folder, f)
                if f.endswith(".rst7") or f.endswith(".inpcrd"):
                    rst7 = os.path.join(folder, f)
            if prmtop and rst7:
                pairs.append((prmtop, rst7))
    return pairs


#################################################################
# CIRCULAR STD
#################################################################
def circular_std(angle_array):
    sin_mean = np.mean(np.sin(angle_array), axis=0)
    cos_mean = np.mean(np.cos(angle_array), axis=0)
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    var = -2.0 * np.log(R)
    var = np.clip(var, 0, None)
    return np.degrees(np.sqrt(var))


#################################################################
# ANALYSIS ONLY
#################################################################
def analyze_only(traj_path, prmtop_path, outdir):
    print(f"=== ANALYZING {traj_path} ===")
    os.makedirs(outdir, exist_ok=True)

    traj = md.load(traj_path, top=prmtop_path)
    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)

    phi_std = circular_std(phi)
    psi_std = circular_std(psi)
    movement = phi_std + psi_std
    residues = [r for (_, r) in phi_idx]

    # φ plot
    plt.figure(figsize=(10,4))
    plt.plot(residues, phi_std)
    plt.title("Phi fluctuations")
    plt.savefig(f"{outdir}/phi_fluctuations.png")
    plt.close()

    # ψ plot
    plt.figure(figsize=(10,4))
    plt.plot(residues, psi_std)
    plt.title("Psi fluctuations")
    plt.savefig(f"{outdir}/psi_fluctuations.png")
    plt.close()

    # Movement
    plt.figure(figsize=(10,4))
    plt.plot(residues, movement)
    plt.title("Movement Score")
    plt.savefig(f"{outdir}/movement_score.png")
    plt.close()

    print(f"=== ANALYSIS COMPLETE → saved to {outdir} ===")


#################################################################
# MD SIMULATION + ANALYSIS
#################################################################
def run_single_sim(index, prmtop_path, rst7_path, args):
    run_name = os.path.splitext(os.path.basename(prmtop_path))[0]
    outdir = f"sim_{run_name}"
    os.makedirs(outdir, exist_ok=True)

    trajout = f"{outdir}/traj.dcd"
    logfile = f"{outdir}/log.txt"

    print(f"=== Starting MD: {run_name} ===")

    prmtop = app.AmberPrmtopFile(prmtop_path)
    inpcrd = app.AmberInpcrdFile(rst7_path)

    system = prmtop.createSystem(
        implicitSolvent=app.OBC2,
        soluteDielectric=1.0,
        solventDielectric=80.0,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.2 * unit.nanometer
    )

    integrator = mm.LangevinIntegrator(
        args.temp * unit.kelvin,
        1 / unit.picosecond,
        args.timestep * unit.picoseconds
    )

    platform = mm.Platform.getPlatformByName("CUDA")

    sim = app.Simulation(prmtop.topology, system, integrator, platform)
    sim.context.setPositions(inpcrd.positions)

    sim.minimizeEnergy()

    sim.reporters.append(app.DCDReporter(trajout, args.interval))
    sim.reporters.append(app.StateDataReporter(
        logfile, args.interval, step=True,
        temperature=True, potentialEnergy=True
    ))

    sim.step(args.steps)

    print(f"MD DONE → running analysis {run_name}")
    analyze_only(trajout, prmtop_path, outdir)


#################################################################
# MAIN
#################################################################
def main():
    args = get_args()

    # --- ANALYSIS-ONLY ---
    if args.analyze_only:
        if not args.traj or not args.prmtop or not args.output_prefix:
            print("ERROR: Must specify --traj --prmtop --output-prefix")
            sys.exit(1)
        analyze_only(args.traj, args.prmtop, args.output_prefix)
        return

    # --- FULL MD MODE ---
    if args.auto:
        pairs = autodetect_inputs()
    else:
        if len(args.inputs) % 2 != 0:
            print("ERROR: --inputs must contain prmtop/rst7 pairs")
            sys.exit(1)
        pairs = [(args.inputs[i], args.inputs[i+1])
                 for i in range(0, len(args.inputs), 2)]

    print("Running simulations:")
    for p, r in pairs:
        print(" -", p, r)

    with Pool(processes=args.parallel) as pool:
        worker = partial(run_single_sim, args=args)
        pool.starmap(worker, [(i+1, p, r) for i, (p, r) in enumerate(pairs)])


if __name__ == "__main__":
    main()
