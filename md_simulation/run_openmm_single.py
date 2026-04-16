#!/usr/bin/env python3
"""
run_openmm_single.py — OpenMM implicit-solvent (GB-OBC2) MD
Resumes from checkpoint and auto-extends existing replica trajectories.

Trajectory naming convention: traj.dcd → traj2.dcd → traj3.dcd → ...
Log naming convention:        log.csv  → log2.csv  → log3.csv  → ...

The script auto-detects the next available name in each sequence so that
running it repeatedly always appends a new segment without manual renaming.

Usage examples:
  # Extend replica 1 by 1 µs (100 000 frames @ 10 ps/frame):
  python run_openmm_single.py \\
      --system_dir /path/to/system \\
      --run_dir    /path/to/run \\
      --replica_id 1 \\
      --target_frames 100000

  # Preview what would happen without running:
  python run_openmm_single.py ... --dry_run

  # Override output name explicitly:
  python run_openmm_single.py ... --out_name traj5.dcd --log_name log5.csv
"""

import argparse
import struct
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Tee: write to both stdout and a file simultaneously
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, filepath: Path):
        self._file   = open(filepath, "w", buffering=1)  # line-buffered
        self._stdout = sys.__stdout__
        self._stderr = sys.__stderr__

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Make it a valid stream for stderr redirect too
    fileno = lambda self: self._stdout.fileno()

import mdtraj as md
from openmm import unit, Platform, LangevinMiddleIntegrator
from openmm.app import (
    AmberPrmtopFile,
    AmberInpcrdFile,
    Simulation,
    DCDReporter,
    StateDataReporter,
    CheckpointReporter,
    NoCutoff,
    HBonds,
    PDBFile,
    OBC2,
)


# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------

def pick_platform(name: str):
    if name == "auto":
        for p in ["CUDA", "OpenCL", "CPU"]:
            try:
                return Platform.getPlatformByName(p), p
            except Exception:
                pass
        raise RuntimeError("No OpenMM platform available (CUDA / OpenCL / CPU)")
    return Platform.getPlatformByName(name), name


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def dcd_nframes_header(path: Path) -> int:
    """Read frame count from DCD header without loading coordinates."""
    with open(path, "rb") as f:
        f.read(4)       # leading block size
        f.read(4)       # 'CORD'
        nframes = struct.unpack("i", f.read(4))[0]
    return nframes


def traj_sequence(outdir: Path) -> list[Path]:
    """Return all existing traj*.dcd files in segment order."""
    segs = []
    for name in ["traj.dcd"] + [f"traj{i}.dcd" for i in range(2, 100)]:
        p = outdir / name
        if p.exists():
            segs.append(p)
        else:
            break
    return segs


def next_traj_name(outdir: Path) -> str:
    """Return the name of the next traj segment to write."""
    existing = traj_sequence(outdir)
    if not existing:
        return "traj.dcd"
    last = existing[-1].name
    if last == "traj.dcd":
        return "traj2.dcd"
    n = int(last.replace("traj", "").replace(".dcd", ""))
    return f"traj{n + 1}.dcd"


def log_sequence(outdir: Path) -> list[Path]:
    """Return all existing log*.csv files in order."""
    logs = []
    for name in ["log.csv"] + [f"log{i}.csv" for i in range(2, 100)]:
        p = outdir / name
        if p.exists():
            logs.append(p)
        else:
            break
    return logs


def next_log_name(outdir: Path) -> str:
    """Return the name of the next log file to write."""
    existing = log_sequence(outdir)
    if not existing:
        return "log.csv"
    last = existing[-1].name
    if last == "log.csv":
        return "log2.csv"
    n = int(last.replace("log", "").replace(".csv", ""))
    return f"log{n + 1}.csv"


def segment_index(outdir: Path) -> int:
    """Return the 1-based index of the segment we are about to write."""
    return len(traj_sequence(outdir)) + 1


def total_existing_ns(outdir: Path) -> float:
    """Sum DCD frame counts × 10 ps/frame across all existing segments."""
    frames = sum(dcd_nframes_header(p) for p in traj_sequence(outdir))
    return frames * 0.01   # 5000 steps × 2 fs = 10 ps per frame → ns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="OpenMM MD (GB-OBC2 implicit solvent) — auto-extend existing replicas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    ap.add_argument("--system_dir", required=True,
                    help="Directory with *_05_minimized.prmtop (and optionally .rst7)")
    ap.add_argument("--run_dir", required=True,
                    help="Parent directory of replica_XX folders")

    # Replica & run length
    ap.add_argument("--replica_id", type=int, required=True)
    ap.add_argument("--target_frames", type=int, required=True,
                    help="Frames to write in this segment (@ --report steps/frame)")

    # Output overrides
    ap.add_argument("--out_name", default=None,
                    help="Override auto-detected traj name (e.g. traj5.dcd)")
    ap.add_argument("--log_name", default=None,
                    help="Override auto-detected log name (e.g. log5.csv)")
    ap.add_argument("--checkpoint_file", default="checkpoint.chk",
                    help="Checkpoint file to resume from (relative to replica dir)")

    # Platform
    ap.add_argument("--platform", default="auto",
                    choices=["auto", "CUDA", "OpenCL", "CPU"])

    # MD parameters
    ap.add_argument("--temp",     type=float, default=310.0,  help="Temperature (K)")
    ap.add_argument("--dt",       type=float, default=2.0,    help="Timestep (fs)")
    ap.add_argument("--friction", type=float, default=1.0,    help="Friction (ps⁻¹)")
    ap.add_argument("--salt",     type=float, default=0.15,   help="Salt conc (M)")

    # Reporting
    ap.add_argument("--report",     type=int, default=5000,
                    help="Steps between DCD frames (5000 × 2 fs = 10 ps/frame)")
    ap.add_argument("--checkpoint", type=int, default=25000,
                    help="Steps between checkpoint writes")
    ap.add_argument("--chunk",      type=int, default=5000,
                    help="Steps per sim.step() call (for interrupt safety)")

    # Misc
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed override (default: 10000 + replica_id × 100 + segment_index)")
    ap.add_argument("--out_file", default=None,
                    help="Override auto-detected .out log path (default: <traj_name>.out in replica dir)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print plan and exit without running MD")

    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    system_dir = Path(args.system_dir).resolve()
    run_dir    = Path(args.run_dir).resolve()
    outdir     = run_dir / f"replica_{args.replica_id:02d}"
    outdir.mkdir(parents=True, exist_ok=True)

    # prmtop (required)
    prmtop_matches = list(system_dir.glob("*_05_minimized.prmtop"))
    if not prmtop_matches:
        sys.exit(f"ERROR: no *_05_minimized.prmtop found in {system_dir}")
    prmtop_path = prmtop_matches[0]

    # rst7 (optional when checkpoint present)
    rst7_matches = list(system_dir.glob("*_05_minimized.rst7"))
    rst7_path = rst7_matches[0] if rst7_matches else None

    # Checkpoint
    chk_path   = outdir / args.checkpoint_file
    final_pdb  = outdir / "final.pdb"

    # Auto-detect output names
    seg_idx  = segment_index(outdir)
    out_name = args.out_name or next_traj_name(outdir)
    log_name = args.log_name or next_log_name(outdir)
    out_dcd  = outdir / out_name
    log_path = outdir / log_name

    # Redirect stdout + stderr to tee (file + terminal)
    out_file_path = Path(args.out_file) if args.out_file else outdir / out_name.replace(".dcd", ".out")
    tee = Tee(out_file_path)
    sys.stdout = tee
    sys.stderr = tee

    # Seed: derive from replica + segment so each segment has a unique seed
    seed = args.seed if args.seed is not None else 10000 + args.replica_id * 100 + seg_idx

    # ------------------------------------------------------------------
    # Pre-flight summary
    # ------------------------------------------------------------------
    existing_ns   = total_existing_ns(outdir)
    new_ns        = args.target_frames * args.report * args.dt * 1e-6  # fs → µs → ns
    existing_segs = traj_sequence(outdir)

    print("\n=== OpenMM MD — segment extension ===")
    print(f"  System          : {system_dir.name}")
    print(f"  Replica         : {args.replica_id:02d}  ({outdir})")
    print(f"  Existing segs   : {len(existing_segs)}  ({existing_ns:.0f} ns = {existing_ns/1000:.3f} µs)")
    for p in existing_segs:
        n = dcd_nframes_header(p)
        print(f"                    {p.name}: {n} frames ({n*0.01:.0f} ns)")
    print(f"  New segment     : {out_name}  →  {args.target_frames} frames ({new_ns:.0f} ns)")
    print(f"  Log             : {log_name}")
    print(f"  Checkpoint      : {chk_path.name}  (exists: {chk_path.exists()})")
    print(f"  stdout/stderr   : {out_file_path}")
    print(f"  Seed            : {seed}  (replica={args.replica_id}, segment={seg_idx})")
    print(f"  Platform        : {args.platform}")
    print(f"  T={args.temp} K  dt={args.dt} fs  friction={args.friction} ps⁻¹  salt={args.salt} M")
    print(f"  After this run  : {existing_ns + new_ns:.0f} ns = {(existing_ns + new_ns)/1000:.3f} µs")

    if out_dcd.exists():
        sys.exit(f"\nERROR: {out_dcd} already exists — refusing to overwrite.\n"
                 f"       Use --out_name to specify a different output name.")

    if not chk_path.exists() and rst7_path is None:
        sys.exit(f"\nERROR: no checkpoint found at {chk_path} and no rst7 available.\n"
                 f"       Cannot determine starting positions.")

    if args.dry_run:
        print("\n[dry_run] Exiting without running MD.")
        return

    # ------------------------------------------------------------------
    # Build OpenMM system
    # ------------------------------------------------------------------
    print("\n[setup] Loading topology...")
    prmtop = AmberPrmtopFile(str(prmtop_path))

    system = prmtop.createSystem(
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
        rigidWater=True,
        implicitSolvent=OBC2,
        implicitSolventSaltConc=args.salt * unit.molar,
    )

    integrator = LangevinMiddleIntegrator(
        args.temp    * unit.kelvin,
        args.friction / unit.picosecond,
        args.dt      * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(seed)

    platform, pname = pick_platform(args.platform)
    props = {}
    if pname == "CUDA":
        props["CudaPrecision"]     = "mixed"
        props["DeterministicForces"] = "true"
    print(f"[setup] Platform: {pname}")

    sim = Simulation(prmtop.topology, system, integrator, platform, props)

    # Set initial positions (overridden by checkpoint if available)
    if rst7_path is not None:
        inpcrd = AmberInpcrdFile(str(rst7_path))
        sim.context.setPositions(inpcrd.positions)

    if chk_path.exists():
        print(f"[resume] Loading checkpoint: {chk_path}")
        sim.context.loadCheckpoint(chk_path.read_bytes())
    else:
        print("[init] No checkpoint — initialising velocities from temperature")
        sim.context.setVelocitiesToTemperature(args.temp * unit.kelvin)

    # ------------------------------------------------------------------
    # Reporters
    # ------------------------------------------------------------------
    sim.reporters.append(DCDReporter(str(out_dcd), args.report, append=False))

    sim.reporters.append(StateDataReporter(
        str(log_path),
        args.report,
        step=True, time=True,
        potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True,
        separator=",",
    ))

    sim.reporters.append(CheckpointReporter(str(chk_path), args.checkpoint))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    steps_total = args.target_frames * args.report
    steps_done  = 0
    print(f"\n[run] {steps_total:,} steps  ({args.target_frames} frames)...")

    while steps_done < steps_total:
        step_now    = min(args.chunk, steps_total - steps_done)
        sim.step(step_now)
        steps_done += step_now

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    state = sim.context.getState(getPositions=True)
    with open(final_pdb, "w") as f:
        PDBFile.writeFile(prmtop.topology, state.getPositions(), f)

    chk_path.write_bytes(sim.context.createCheckpoint())

    frames_written = dcd_nframes_header(out_dcd)
    print(f"\n[done] {out_name}: {frames_written} frames ({frames_written * 0.01:.0f} ns)")
    print(f"       Total replica time: {existing_ns + frames_written * 0.01:.0f} ns "
          f"= {(existing_ns + frames_written * 0.01) / 1000:.3f} µs")
    tee.close()


if __name__ == "__main__":
    main()
