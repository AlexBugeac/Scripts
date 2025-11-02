import sys
import argparse
import numpy as np
import mdtraj as md
import os
import contextlib
import time

# Suppress OpenMM verbose output
os.environ['OPENMM_PLUGIN_DIR'] = '/dev/null'

sys.path.append("/home/alexb/Robosample/bin")
sys.path.append("/home/laurentiu/git6/Robosample/bin")
import flexor3 as flexor
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
    "Slider": robosample.BondMobility.Translation
}

def getFlexibilitiesFromFile(flexFN):
    flexibilities = []
    with open(flexFN, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] != "Weld":
                aIx_1, aIx_2, mobilityStr = int(parts[0]), int(parts[1]), parts[2]
                mobility = mobilityMap[mobilityStr]
                flexibilities.append(robosample.BondFlexibility(aIx_1, aIx_2, mobility))
    return flexibilities

def run_single_folding_simulation(run_id, args, extended_rst7):
    """Run a single folding simulation with minimal output"""
    
    # Create unique name for this run
    run_name = f"{args.name}_fold_{run_id + 1:03d}"
    
    # Use different random seed for each run
    run_seed = args.seed + run_id * 1000
    
    run_type = getattr(robosample.RunType, args.runType)
    context = robosample.Context(run_name, run_seed, 0, 1, run_type, 1, 0)
    context.setPdbRestartFreq(0)
    context.setPrintFreq(args.writeFreq)
    context.setNonbonded(0, 1.2)
    context.setGBSA(1)
    context.setVerbose(False)  # Disable verbose output
    context.loadAmberSystem(args.top, extended_rst7)
    
    # Create flexibility objects
    with suppress_stdout_stderr():
        mdtrajObj = md.load(extended_rst7, top=args.top)
        flexorObj = flexor.Flexor(mdtrajObj)
    nofWorlds = 0
    
    # World 0: Full Cartesian (main folding dynamics)
    flex_Cart = flexorObj.create(range="all", subset=["all"], jointType="Cartesian")
    context.addWorld(False, 1, robosample.RootMobility.CARTESIAN, flex_Cart, True, False, 0)
    nofWorlds += 1
    
    # World 1: Enhanced backbone sampling for folding
    flex_Roll = flexorObj.create(range="all", distanceCutoff=0.45, subset=["rama"], jointType="Pin", sasa_value=1.5)
    context.addWorld(False, 1, robosample.RootMobility.WELD, flex_Roll, True, False, 0)
    context.getWorld(nofWorlds).setRollFlexibilities(True)
    nofWorlds += 1
    
    # World 2: Ramachandran sampling (important for folding)
    flex_Rama = flexorObj.create(range="all", distanceCutoff=0, subset=["rama"], jointType="Pin", sasa_value=-1.0)
    context.addWorld(False, 1, robosample.RootMobility.WELD, flex_Rama, True, False, 0)
    nofWorlds += 1
    
    # Additional flexibilities from file
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
    
    # Single replica at target temperature (faster folding)
    temperatures = np.array([args.baseTemperature], dtype=np.float64)
    
    accept_reject_modes = [robosample.AcceptRejectMode.MetropolisHastings] * nofWorlds
    # Faster timesteps for folding dynamics
    timesteps = [0.0005, 0.05, 0.005, 0.002, 0.002]
    # Fewer MD steps per move for faster transitions
    mdsteps = [2000, 5, 8, 10, 10]
    integrators = [robosample.IntegratorType.OMMVV] + [robosample.IntegratorType.VERLET] * (nofWorlds - 1)
    distort_options = [0] * nofWorlds
    distort_args = ["0"] * nofWorlds
    flow = [0] * nofWorlds
    work = [0] * nofWorlds
    
    context.addReplica(0)
    context.addThermodynamicState(
        0,
        temperatures[0],
        accept_reject_modes,
        distort_options,
        distort_args,
        flow,
        work,
        integrators,
        list(range(nofWorlds)),
        timesteps,
        mdsteps
    )
    
    with suppress_stdout_stderr():
        context.Initialize()
    
    # Only REX output will be produced
    context.RunREX(args.equilSteps, args.foldingSteps)
    
    return run_name

parser = argparse.ArgumentParser(description="Folding simulation - multiple independent runs")
parser.add_argument('--name', type=str, required=True, help='Base simulation name')
parser.add_argument('--top', type=str, required=True, help='Topology file (.prmtop)')
parser.add_argument('--rst7', type=str, required=True, help='Starting structure (.rst7) - preferably extended for folding')
parser.add_argument('--numRuns', type=int, default=10, help='Number of independent folding runs')
parser.add_argument('--equilSteps', type=int, default=10000, help='Brief equilibration steps')
parser.add_argument('--foldingSteps', type=int, default=500000, help='Folding simulation steps per run')
parser.add_argument('--prodSteps', type=int, help='Alias for foldingSteps (for compatibility)')
parser.add_argument('--writeFreq', type=int, default=50, help='Output frequency')
parser.add_argument('--baseTemperature', type=float, default=300.0, help='Folding temperature (K)')
parser.add_argument('--runType', type=str, default='DEFAULT', help='Run type')
parser.add_argument('--seed', type=int, default=12345, help='Base random seed')
parser.add_argument('--flexFNs', type=str, nargs='+', default=[], help='Flexibility files')
parser.add_argument('--parallel', action='store_true', help='Run simulations in parallel (experimental)')
args = parser.parse_args()

# Handle prodSteps alias for compatibility
if args.prodSteps is not None:
    args.foldingSteps = args.prodSteps

# Run simulations sequentially with minimal output
for run_id in range(args.numRuns):
    try:
        run_name = run_single_folding_simulation(run_id, args, args.rst7)
        # Brief pause between runs to allow file system sync
        time.sleep(2)
    except Exception as e:
        # Silent error handling - just continue to next run
        continue