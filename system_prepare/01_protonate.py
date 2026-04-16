#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import tempfile
import shutil

# ---------------------------------------------------------------
# Helper: Run commands
# ---------------------------------------------------------------
def run_cmd(cmd):
    print(">>>", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError("Command failed: " + " ".join(cmd))

# ---------------------------------------------------------------
# Helper: Detect HIS state from pdb2pqr output
# ---------------------------------------------------------------
def detect_histidine_state(atom_names):
    has_HD1 = any(a.strip() == "HD1" for a in atom_names)
    has_HE2 = any(a.strip() == "HE2" for a in atom_names)

    if has_HD1 and has_HE2:
        return "HIP"
    if has_HD1:
        return "HID"
    if has_HE2:
        return "HIE"
    return "HIS"

# ---------------------------------------------------------------
# Extract HIS protonation from pdb2pqr PDB
# ---------------------------------------------------------------
def read_histidine_states_from_pdb2pqr(pdb_path):
    """
    Return a dict: (chain, resid) -> HID/HIE/HIP
    """
    residues = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            if line[17:20].strip() != "HIS":
                continue

            chain = line[21]
            resid = line[22:26].strip()
            atom = line[12:16]

            key = (chain, resid)
            residues.setdefault(key, []).append(atom)

    # Convert to states
    states = {k: detect_histidine_state(v) for k, v in residues.items()}
    return states

# ---------------------------------------------------------------
# Rewrite HIS names in original PDB (no hydrogens touched)
# ---------------------------------------------------------------
def apply_histidine_states_to_original(original_pdb, output_pdb, his_states):
    with open(original_pdb) as fin, open(output_pdb, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                if line[17:20].strip() == "HIS":
                    key = (line[21], line[22:26].strip())
                    new = his_states.get(key, "HIS")
                    line = line[:17] + new.ljust(3) + line[20:]
            fout.write(line)

# ---------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------
def run_protonate(input_pdb, output_pdb, ph=7.4):
    """
    1. Run pdb2pqr → tmp.pdb (adds H)
    2. Read HIS protonation state from tmp.pdb
    3. Rewrite HIS only in ORIGINAL PDB (no hydrogens)
    """
    tmpdir = tempfile.mkdtemp()
    tmp_pdb = Path(tmpdir) / "pqr_output.pdb"
    tmp_pqr = Path(tmpdir) / "pqr_output.pqr"

    # --- Step 1: run pdb2pqr ---
    run_cmd([
        "pdb2pqr",
        "--ff=AMBER",
        "--with-ph", str(ph),
        "--keep-chain",
        "--pdb-output", str(tmp_pdb),
        input_pdb,
        str(tmp_pqr)
    ])

    print("[+] Reading HIS protonation from pdb2pqr output…")
    his_states = read_histidine_states_from_pdb2pqr(tmp_pdb)

    print("[+] Applying HIS states to original PDB…")
    apply_histidine_states_to_original(input_pdb, output_pdb, his_states)

    shutil.rmtree(tmpdir)

    print(f"[✓] Protonation complete → {output_pdb}")
    print("[✓] HIS/HID/HIE/HIP updated, no hydrogens added")

# ---------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_pdb")
    p.add_argument("output_pdb")
    p.add_argument("--ph", type=float, default=7.4)
    a = p.parse_args()

    run_protonate(a.input_pdb, a.output_pdb, a.ph)

if __name__ == "__main__":
    main()
