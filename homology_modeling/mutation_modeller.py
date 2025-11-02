#!/usr/bin/env python3
"""
Protein Mutation and Disulfide Bond Modeling Script
Mutates residues and enforces disulfide bond constraints using MODELLER
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add MODELLER to path
sys.path.append('/home/alexb/miniconda3/envs/modeller/lib/modeller-10.7/modlib')

from modeller import *
from modeller.automodel import *
from modeller.selection import Selection

def setup_logging(output_dir, verbose=False):
    """Setup logging configuration"""
    log_file = Path(output_dir) / "mutation_modeller.log"
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class MutationAutoModel(AutoModel):
    """Custom AutoModel class for mutations and disulfide bonds"""
    
    def __init__(self, env, alnfile, knowns, sequence, mutations=None, disulfide_pairs=None, **kwargs):
        super().__init__(env, alnfile, knowns, sequence, **kwargs)
        self.mutations = mutations or []
        self.disulfide_pairs = disulfide_pairs or []
        self.logger = logging.getLogger(__name__)
        
    def select_atoms(self):
        """Select atoms for optimization"""
        return Selection(self)
    
    def special_restraints(self, aln):
        """Add disulfide bond restraints"""
        if not self.disulfide_pairs:
            return
            
        self.logger.info(f"Adding {len(self.disulfide_pairs)} disulfide bond restraints")
        
        for res1, res2 in self.disulfide_pairs:
            self.logger.info(f"  Disulfide bond: Cys{res1} ↔ Cys{res2}")
            
            # Add distance restraint between sulfur atoms
            self.restraints.add(
                Forms.Gaussian(group=Physical.xy_distance,
                              feature=Features.Distance(
                                  self.residue_range(f'{res1}:').atom['SG'],
                                  self.residue_range(f'{res2}:').atom['SG']
                              ),
                              mean=2.05,    # Typical S-S bond length in Å
                              stdev=0.1)    # Tight constraint
            )

def create_mutation_alignment(template_pdb, mutations, output_pir, logger):
    """Create PIR alignment file for mutations"""
    
    logger.info(f"Creating mutation alignment for: {mutations}")
    
    # Read the template PDB to get sequence
    env = Environ()
    env.io.atom_files_directory = [str(Path(template_pdb).parent)]
    
    mdl = Model(env, file=Path(template_pdb).name)
    
    # Get sequence from model
    template_seq = ""
    for residue in mdl.residues:
        template_seq += residue.code
    
    logger.info(f"Template sequence length: {len(template_seq)}")
    logger.info(f"First 50 residues: {template_seq[:50]}")
    
    # Apply mutations to create target sequence
    target_seq = list(template_seq)
    
    for pos, new_aa in mutations:
        if 1 <= pos <= len(target_seq):
            old_aa = target_seq[pos-1]  # Convert to 0-based
            target_seq[pos-1] = new_aa
            logger.info(f"  Mutation {pos}: {old_aa} → {new_aa}")
        else:
            logger.warning(f"  Invalid position {pos} (sequence length: {len(target_seq)})")
    
    target_seq_str = "".join(target_seq)
    
    # Write PIR alignment
    template_name = Path(template_pdb).stem
    
    with open(output_pir, 'w') as f:
        # Target sequence (mutated)
        f.write(f">P1;target_mutated\n")
        f.write(f"sequence:target_mutated:1:A:{len(target_seq_str)}:A:Mutated protein: : 0.00: 0.00\n")
        f.write(f"{target_seq_str}*\n")
        
        # Template structure
        f.write(f">P1;{template_name}\n")
        f.write(f"structureX:{template_name}:1:A:{len(template_seq)}:A:Template protein: : 2.00: 0.19\n")
        f.write(f"{template_seq}*\n")
    
    logger.info(f"Mutation alignment saved to: {output_pir}")
    return output_pir, template_name

def run_mutation_modeling(template_pdb, mutations, disulfide_pairs, output_dir, num_models=3, logger=None):
    """Run MODELLER for mutation modeling with disulfide constraints"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy template to output directory
    template_copy = output_dir / Path(template_pdb).name
    if not template_copy.exists():
        import shutil
        shutil.copy2(template_pdb, template_copy)
        logger.info(f"Copied template to: {template_copy}")
    
    # Create alignment file
    pir_file = output_dir / "mutation_alignment.pir"
    pir_path, template_name = create_mutation_alignment(template_pdb, mutations, pir_file, logger)
    
    # Set up MODELLER environment
    env = Environ()
    env.io.atom_files_directory = [str(output_dir)]
    
    # Run mutation modeling
    logger.info("Starting mutation modeling with MODELLER...")
    
    a = MutationAutoModel(
        env,
        alnfile=str(pir_file),
        knowns=[template_name],
        sequence='target_mutated',
        mutations=mutations,
        disulfide_pairs=disulfide_pairs
    )
    
    a.starting_model = 1
    a.ending_model = num_models
    
    # Generate models
    a.make()
    
    # Rename output models
    logger.info("Renaming output models...")
    for i in range(1, num_models + 1):
        old_name = output_dir / f"target_mutated.B9999000{i}.pdb"
        new_name = output_dir / f"mutated_model_{i:02d}.pdb"
        
        if old_name.exists():
            old_name.rename(new_name)
            logger.info(f"Renamed {old_name.name} -> {new_name.name}")
    
    logger.info(f"Mutation modeling completed! Models saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate mutated protein models with disulfide bonds")
    parser.add_argument('-i', '--input-pdb', required=True, help='Input template PDB file')
    parser.add_argument('-o', '--output-dir', required=True, help='Output directory')
    parser.add_argument('-m', '--mutations', required=True, help='Mutations as "pos1:old>new,pos2:old>new" (e.g., "28:I>C,41:S>C")')
    parser.add_argument('-d', '--disulfide', help='Disulfide pairs as "res1-res2,res3-res4" (e.g., "41-46")')
    parser.add_argument('-n', '--num-models', type=int, default=3, help='Number of models to generate')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.verbose)
    
    logger.info("=== MUTATION MODELING WORKFLOW ===")
    logger.info(f"Input template: {args.input_pdb}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Parse mutations
    mutations = []
    if args.mutations:
        for mut_str in args.mutations.split(','):
            parts = mut_str.strip().split(':')
            if len(parts) == 2:
                pos = int(parts[0])
                old_new = parts[1].split('>')
                if len(old_new) == 2:
                    mutations.append((pos, old_new[1]))
                    logger.info(f"Mutation: position {pos} -> {old_new[1]}")
    
    # Parse disulfide bonds
    disulfide_pairs = []
    if args.disulfide:
        for pair_str in args.disulfide.split(','):
            parts = pair_str.strip().split('-')
            if len(parts) == 2:
                res1, res2 = int(parts[0]), int(parts[1])
                disulfide_pairs.append((res1, res2))
                logger.info(f"Disulfide bond: {res1}-{res2}")
    
    if not mutations:
        logger.error("No valid mutations specified!")
        return
    
    # Run modeling
    try:
        run_mutation_modeling(
            args.input_pdb,
            mutations,
            disulfide_pairs,
            args.output_dir,
            args.num_models,
            logger
        )
        logger.info("=== MUTATION MODELING COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Mutation modeling failed: {e}")
        raise

if __name__ == "__main__":
    main()