#!/usr/bin/env python3
import subprocess
from pathlib import Path

def run_cmd(cmd, desc=""):
    print(f">>> {' '.join(cmd)}")
    if desc:
        print(f"[+] {desc}")
    subprocess.run(cmd, check=True)


class SystemCapping:
    """
    Handles:
    - Capping with ACE/NME
    - Generating capped.prmtop / capped.rst7
    - Generating final PRMTOP/RST7 with SSBOND after minimization
    """

    def __init__(self, prefix):
        self.prefix = prefix


    # ============================================================
    #  STEP 4 — BUILD INITIAL CAPPED SYSTEM USING TRIMMED INPUT
    # ============================================================
    def run_capping(self, capped_input, ss_pairs):
        """
        Build capped system from the *trimmed* or *untrimmed* input PDB.
        (We no longer reference _02_cyx.pdb!)
        """

        with open("leap.in", "w") as f:
            f.write("source leaprc.protein.ff14SB\n")
            f.write(f"mol = loadPdb \"{capped_input}\"\n")
            f.write("check mol\n")
            f.write("saveamberparm mol capped.prmtop capped.rst7\n")
            f.write("savepdb mol capped.pdb\n")
            f.write("quit\n")

        run_cmd(["tleap", "-f", "leap.in"], desc="Building capped system")

        print("[✓] Capped system built: capped.prmtop / capped.rst7 / capped.pdb")
        print("[+] Minimization will run next…")
        print("[+] After minimization, final LEaP SSBOND step will be executed.")


    # ============================================================
    #  STEP 6 — FINAL LEaP WITH SSBOND USING MINIMIZED PDB
    # ============================================================
    def write_final_leap(self, minimized_pdb, ss_pairs):
        """
        Apply SSBOND pairs AFTER minimization.
        Uses the MINIMIZED PDB (which was generated from trimmed input).
        """

        final_in = "final_leap.in"

        with open(final_in, "w") as f:
            f.write("source leaprc.protein.ff14SB\n")
            f.write(f"mol = loadPdb \"{minimized_pdb}\"\n\n")

            f.write("### APPLYING DISULFIDE BONDS ###\n")
            for a, b in ss_pairs:
                f.write(f"bond mol.{a}.SG mol.{b}.SG\n")

            f.write("\ncheck mol\n")
            f.write(f"saveamberparm mol {self.prefix}_final.prmtop {self.prefix}_final.rst7\n")
            f.write(f"savepdb mol {self.prefix}_final.pdb\n")
            f.write("quit\n")

        print(f"[✓] Wrote final LEaP file: {final_in}")

        run_cmd(["tleap", "-f", final_in], desc="Generating final PRMTOP/RST7 with SSBOND")
