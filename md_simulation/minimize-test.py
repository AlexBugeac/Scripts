#!/usr/bin/env python3
import subprocess
from openmm.app import (
    AmberPrmtopFile,
    AmberInpcrdFile,
    Simulation,
    PDBReporter,
    StateDataReporter,
    OBC2,
    CutoffNonPeriodic
)
from openmm import LangevinIntegrator
from openmm.unit import kelvin, picosecond, nanometer
import sys
from pathlib import Path

# ============================
# USER INPUTS
# ============================
prmtop_file = "/home/alex/Alex/Proiecte/2025/Cantavac/E2-HCV/final/Modele/Modele_create/Test-open-mm/final_folder/model_8RJJ_mutated41-46_pipeline/model_8RJJ_mutated41-46_05_minimized.prmtop"
rst7_file   = "/home/alex/Alex/Proiecte/2025/Cantavac/E2-HCV/final/Modele/Modele_create/Test-open-mm/final_folder/model_8RJJ_mutated41-46_pipeline/capped.rst7"

output_prefix = "minimized_output"


# ============================
# LOAD AMBER INPUT
# ============================
print("[+] Loading AMBER inputs...")
prmtop = AmberPrmtopFile(prmtop_file)
inpcrd = AmberInpcrdFile(rst7_file)

# ============================
# BUILD SYSTEM
# ============================
system = prmtop.createSystem(
    implicitSolvent=OBC2,
    soluteDielectric=1.0,
    solventDielectric=80.0,
    nonbondedMethod=CutoffNonPeriodic,
    nonbondedCutoff=1.2 * nanometer
)

integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picosecond)
simulation = Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)

# ============================
# MINIMIZATION
# ============================
print("[+] Minimizing energy...")
simulation.minimizeEnergy(maxIterations=5000)

min_pdb = f"{output_prefix}.pdb"
print(f"[+] Writing minimized PDB → {min_pdb}")
simulation.reporters.append(PDBReporter(min_pdb, 1))
simulation.step(1)

# ============================
# RUN TLEAP TO GENERATE FINAL AMBER FILES
# ============================
final_prmtop = f"{output_prefix}.prmtop"
final_rst7   = f"{output_prefix}.rst7"

leapin = f"{output_prefix}_tleap.in"
with open(leapin, "w") as f:
    f.write(f"""
source leaprc.protein.ff14SB

mol = loadpdb {min_pdb}

saveamberparm mol {final_prmtop} {final_rst7}

quit
""")

print("[+] Running tleap to write final amber files...")
subprocess.run(["tleap", "-f", leapin], check=True)

print()
print("[✓] Done!")
print(f"[✓] Final PRMTOP: {final_prmtop}")
print(f"[✓] Final RST7:   {final_rst7}")
