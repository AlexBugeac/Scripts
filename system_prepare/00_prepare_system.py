#!/usr/bin/env python3
import argparse
import importlib
import os
from pathlib import Path
import sys

# SIMULATION_ROOT is set from --output-root CLI arg (see main())

# ===========================================================
# Import pipeline modules (UNCHANGED)
# ===========================================================
proton = importlib.import_module("01_protonate")
cyx    = importlib.import_module("02_detect_cyx")
trim   = importlib.import_module("03_trim_residues")
capmod = importlib.import_module("04_build_capped_system")
minim  = importlib.import_module("05_minimize")

# ===========================================================
# CANONICAL DISULFIDES (NATIVE)
# ===========================================================
NATIVE_SSBONDS = [
    (46, 120),
    (69, 237),
    (76, 103),
    (111, 181),
    (125, 169),
    (186, 214),
    (198, 202),
    (224, 261),
]

# ===========================================================
# Decide disulfide topology (SINGLE SOURCE OF TRUTH)
# ===========================================================
def decide_ssbonds(model_name, cli_ssbonds):
    if cli_ssbonds:
        it = iter(cli_ssbonds)
        return list(zip(it, it))

    if "db411-429" in model_name:
        return [
            (411, 429),
            (69, 237),
            (76, 103),
            (111, 181),
            (125, 169),
            (186, 214),
            (198, 202),
            (224, 261),
        ]

    if "db411-424" in model_name:
        return [
            (411, 424),
            (69, 237),
            (76, 103),
            (111, 181),
            (125, 169),
            (186, 214),
            (198, 202),
            (224, 261),
        ]

    # Default: native-like topology
    return NATIVE_SSBONDS


# ===========================================================
# Process ONE model
# ===========================================================
def process_single_pdb(input_pdb, ph, cli_ssbonds, trim_ranges, min_steps):
    input_path = Path(input_pdb).resolve()
    base = input_path.stem

    outdir = SIMULATION_ROOT / f"{base}_pipeline"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n[+] Processing {base}")
    print(f"[+] Output → {outdir}")

    os.chdir(outdir)

    # -------------------------------------------------------
    # 1. Protonation
    # -------------------------------------------------------
    p1 = f"{base}_01_protonated.pdb"
    proton.run_protonate(str(input_path), p1, ph)

    # -------------------------------------------------------
    # 2. Decide disulfides
    # -------------------------------------------------------
    ss_pairs = decide_ssbonds(base, cli_ssbonds)
    print(f"[+] Disulfide topology: {ss_pairs}")

    # -------------------------------------------------------
    # 3. CYX assignment
    # -------------------------------------------------------
    p2 = f"{base}_02_cyx.pdb"
    cyx.run_detect_cyx(p1, p2, ss_pairs)

    # -------------------------------------------------------
    # 4. Trimming (optional)
    # -------------------------------------------------------
    if trim_ranges:
        p3 = f"{base}_03_trimmed.pdb"
        mapping = trim.run_trim(p2, p3, trim_ranges)

        ss_pairs = [
            (mapping[a], mapping[b])
            for (a, b) in ss_pairs
            if a in mapping and b in mapping
        ]
        capped_input = p3
    else:
        capped_input = p2

    # -------------------------------------------------------
    # 5. Capping + LEaP
    # -------------------------------------------------------
    cap = capmod.SystemCapping(base)
    cap.run_capping(capped_input, ss_pairs)

    # -------------------------------------------------------
    # 6. Minimization
    # -------------------------------------------------------
    min_prefix = f"{base}_05_minimized"
    minim.run_minimize("capped.prmtop", "capped.rst7", min_prefix, min_steps)

    # -------------------------------------------------------
    # 7. Final LEaP
    # -------------------------------------------------------
    cap.write_final_leap(f"{min_prefix}.pdb", ss_pairs)

    print(f"[✓] Finished {base}")


# ===========================================================
# Main
# ===========================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_path", help="Directory containing input PDB files")
    p.add_argument("--output-root", default=None,
                   help="Root output directory (default: same directory as input_path)")
    p.add_argument("--ph", type=float, default=7.4)
    p.add_argument("--ssbond", nargs="+", type=int)
    p.add_argument("--trim", nargs="+")
    p.add_argument("--min_steps", type=int, default=5000)
    a = p.parse_args()

    global SIMULATION_ROOT
    SIMULATION_ROOT = Path(a.output_root).resolve() if a.output_root \
                      else Path(a.input_path).resolve()
    SIMULATION_ROOT.mkdir(parents=True, exist_ok=True)

    root = Path(a.input_path).resolve()
    pdbs = [
        p for p in root.rglob("*.pdb")
        if "test" not in p.parts
    ]

    if not pdbs:
        print("[ERROR] No PDBs found.")
        sys.exit(1)

    print(f"[+] Found {len(pdbs)} models")

    for pdb in sorted(pdbs):
        process_single_pdb(pdb, a.ph, a.ssbond, a.trim, a.min_steps)

    print("\n[✓] ALL MODELS PROCESSED\n")


if __name__ == "__main__":
    main()
