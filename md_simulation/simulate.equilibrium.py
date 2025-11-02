import sys
import argparse
import numpy as np
import mdtraj as md
import os
import contextlib

sys.path.append("/home/alexb/Robosample/bin")
sys.path.append("/home/alexb/Robosample")
sys.path.append("/home/laurentiu/git6/Robosample/bin")
import flexor
import robosample

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

mobilityMap = {
    "Cartesian": robosample.BondMobility.Translation,
    "Pin": robosample.BondMobility.Torsion,
    "Torsion": robosample.BondMobility.Torsion,
    "Slider": robosample.BondMobility.Slider,
}

def getFlexibilitiesFromFile(flexFN):
    flexibilities = []
    with open(flexFN, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] != "Weld":
                aIx_1, aIx_2, mobilityStr = int(parts[0]), int(parts[1]), parts[2]
                mobility = mobilityMap[mobilityStr]
                flexibilities.append(robosample.BondFlexibility(aIx_1, aIx_2, mobility))
    return flexibilities

parser = argparse.ArgumentParser(description="Equilibrium simulation with enhanced sampling")
parser.add_argument('--name', type=str, required=True, help='Simulation name')
parser.add_argument('--top', type=str, required=True, help='Topology file (.prmtop)')
parser.add_argument('--rst7', type=str, required=True, help='Starting structure (.rst7)')
parser.add_argument('--equilSteps', type=int, default=50000, help='Equilibration steps')
parser.add_argument('--prodSteps', type=int, default=5000000, help='Production steps')
parser.add_argument('--writeFreq', type=int, default=50, help='Output frequency')
parser.add_argument('--baseTemperature', type=float, default=300.0, help='Base temperature (K)')
parser.add_argument('--runType', type=str, default='DEFAULT', help='Run type')
parser.add_argument('--seed', type=int, default=12345, help='Random seed')
parser.add_argument('--flexFNs', type=str, nargs='+', default=[], help='Flexibility files')
parser.add_argument('--numReplicas', type=int, default=4, help='Number of replicas for REX')
parser.add_argument('--tempFactor', type=float, default=1.05, help='Temperature scaling factor')
args = parser.parse_args()

run_type = getattr(robosample.RunType, args.runType)
context = robosample.Context(args.name, args.seed, 0, 1, run_type, 1, 0)
context.setPdbRestartFreq(0)
context.setPrintFreq(args.writeFreq)
context.setNonbonded(0, 1.2)
context.setGBSA(1)
context.setVerbose(True)
context.loadAmberSystem(args.top, args.rst7)

# Create flexibility objects
mdtrajObj = md.load(args.rst7, top=args.top)
flexorObj = flexor.Flexor(mdtrajObj)
nofWorlds = 0

# World 0: Full Cartesian
flexes_Cart = flexorObj.create(range="all", subset=["all"], jointType="Cartesian")
context.addWorld(False, 1, robosample.RootMobility.CARTESIAN, flexes_Cart, True, False, 0)
nofWorlds += 1

# World 1: Backbone Roll moves
flex_Roll = flexorObj.create(range="(resid 1 to 47) or (resid 111 to 136)", distanceCutoff=0, subset=["rama"], jointType="Pin", sasa_value=-1.0)
context.addWorld(False, 1, robosample.RootMobility.WELD, flex_Roll, True, False, 0)
context.getWorld(nofWorlds).setRollFlexibilities(True)
nofWorlds += 1

# World 2: Ramachandran sampling
flex_Rama = flexorObj.create(range="all", distanceCutoff=0, subset=["rama"], jointType="Pin", sasa_value=-1.0)
context.addWorld(False, 1, robosample.RootMobility.WELD, flex_Rama, True, False, 0)
nofWorlds += 1

# World 3: Sidechain sampling
flex_Side = flexorObj.create(range="not (resname CYS)", distanceCutoff=0, subset=["side"], jointType="Pin", sasa_value=-1.0)
context.addWorld(False, 10, robosample.RootMobility.WELD, flex_Side, True, False, 0)
nofWorlds += 1

for flexFN in args.flexFNs:
    flexibilities = getFlexibilitiesFromFile(flexFN)
    context.addWorld(False, 1, robosample.RootMobility.WELD, flexibilities, True, False, 0)
    nofWorlds += 1

# Add samplers to each world
sampler = robosample.SamplerName.HMC
thermostat = robosample.ThermostatName.ANDERSEN

context.getWorld(0).addSampler(sampler, robosample.IntegratorType.OMMVV, thermostat, False)
for wIx in range(1, nofWorlds):
    context.getWorld(wIx).addSampler(sampler, robosample.IntegratorType.VERLET, thermostat, True)

# Set up replica exchange
base_temp = args.baseTemperature
num_replicas = 1
temperatures = np.zeros(num_replicas, dtype=np.float64)
for replIx in range(num_replicas):
    temperatures[replIx] = base_temp + (replIx * 10)

accept_reject_modes = nofWorlds * [robosample.AcceptRejectMode.MetropolisHastings]
timesteps = nofWorlds * [0.0007]
timesteps[1] = 0.2  # roll
timesteps[2] = 0.007  # Rama  
timesteps[3] = 0.004  # side
if nofWorlds > 4:
    timesteps[4] = 0.002  # additional

worldIndexes = list(range(nofWorlds))
mdsteps = nofWorlds * [10]
integrators = [robosample.IntegratorType.OMMVV] + ((nofWorlds - 1) * [robosample.IntegratorType.VERLET])
distort_options = nofWorlds * [0]
distort_args = nofWorlds * ["0"]
flow = nofWorlds * [0]
work = nofWorlds * [0]

for replIx in range(num_replicas):
    context.addReplica(replIx)
    context.addThermodynamicState(
        replIx,
        temperatures[replIx],
        accept_reject_modes,
        distort_options,
        distort_args,
        flow,
        work,
        integrators,
        worldIndexes,
        timesteps,
        mdsteps
    )

# Initialize
context.Initialize()

# Run
context.RunREX(args.equilSteps, args.prodSteps)