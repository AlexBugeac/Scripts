#!/usr/bin/env python3
import subprocess
import sys

# ============================================================
# Helper: run shell commands
# ============================================================
def run(cmd, desc=None):
    if desc:
        print(f"[+] {desc}")
    print(">>>", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"ERROR: Command failed → {' '.join(cmd)}")

# ============================================================
# Write AMBER minimization input
# ============================================================
def write_min_in(filename, steps, use_gb=False):
    """
    Generate a safe minimization input for a NON-PERIODIC system.
    GB is optional.
    """

    if use_gb:
        gb_part = "  igb=8,\n  cut=999.0,\n"
    else:
        gb_part = "  cut=10.0,\n"

    with open(filename, "w") as f:
        f.write(f"""Minimize system
&cntrl
  imin=1,
  ntb=0,
  maxcyc={steps},
  ncyc={steps//2},
{gb_part}/
""")

# ============================================================
# Core minimization routine
# ============================================================
def minimize(prmtop_path, rst7_path, prefix, steps, use_gb=False):
    print("[+] Starting AMBER minimization pipeline")

    inp = f"{prefix}_min.in"
    out_rst7 = f"{prefix}.rst7"
    out_pdb = f"{prefix}.pdb"
    log = f"{prefix}_min.out"

    # 1) Write input file
    write_min_in(inp, steps, use_gb=use_gb)

    # 2) Run sander
    run([
        "sander",
        "-O",
        "-i", inp,
        "-p", prmtop_path,
        "-c", rst7_path,
        "-o", log,
        "-r", out_rst7,
        "-ref", rst7_path
    ], desc=f"Minimizing structure for {steps} steps")

    # 3) Convert RST7 → PDB
    with open(out_pdb, "w") as f:
        subprocess.run(
            ["ambpdb", "-p", prmtop_path, "-c", out_rst7],
            stdout=f,
            check=True
        )
    print(f"[+] Wrote minimized PDB → {out_pdb}")

    print("[✓] Minimization complete.")
    print(f"[✓] PRMTOP  : {prmtop_path}")
    print(f"[✓] RST7    : {out_rst7}")
    print(f"[✓] PDB     : {out_pdb}")

    return out_pdb, out_rst7

# ============================================================
# Entry point required by 00_prepare_system.py
# ============================================================
def run_minimize(prmtop, rst7, prefix, steps):
    return minimize(prmtop, rst7, prefix, steps, use_gb=False)

# ============================================================
# Manual usage
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: 05_minimize.py prmtop rst7 prefix steps")
        sys.exit(1)

    prmtop = sys.argv[1]
    rst7 = sys.argv[2]
    prefix = sys.argv[3]
    steps = int(sys.argv[4])

    minimize(prmtop, rst7, prefix, steps)
