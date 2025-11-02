#!/usr/bin/env python3
"""
Regional Hybrid Modeling Approach
Creates separate models for different regions and combines them
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def setup_logging(verbose_level=1):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if verbose_level == 0:
        level = logging.WARNING
    elif verbose_level == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    
    logging.basicConfig(level=level, format=log_format)
    return logging.getLogger(__name__)

def create_region_specific_alignment(original_pir, region_templates, output_file, logger):
    """Create alignment using only specific templates for modeling"""
    
    logger.info(f"Creating region-specific alignment: {output_file}")
    logger.info(f"Using templates: {region_templates}")
    
    # Read original alignment
    with open(original_pir, 'r') as f:
        content = f.read()
    
    # Parse sequences
    sequences = {}
    current_id = None
    lines = content.strip().split('\n')
    
    for line in lines:
        if line.startswith('>P1;'):
            current_id = line[4:]  # Remove '>P1;'
            sequences[current_id] = {'header': '', 'sequence': ''}
        elif current_id and line and not line.startswith('>'):
            if ':' in line and sequences[current_id]['header'] == '':
                sequences[current_id]['header'] = line
            else:
                sequences[current_id]['sequence'] += line.replace('*', '')
    
    # Write new alignment with only target + specified templates
    with open(output_file, 'w') as f:
        # Always include target first
        target_id = 'E2_target'
        f.write(f">P1;{target_id}\n")
        f.write(f"{sequences[target_id]['header']}\n")
        f.write(f"{sequences[target_id]['sequence']}*\n")
        
        # Include specified templates
        for template in region_templates:
            if template in sequences:
                f.write(f">P1;{template}\n")
                f.write(f"{sequences[template]['header']}\n")
                f.write(f"{sequences[template]['sequence']}*\n")
                logger.info(f"  Added template: {template}")
    
    logger.info(f"Regional alignment saved: {output_file}")
    return output_file

def run_modeller_for_region(pir_file, template_files, output_dir, region_name, logger):
    """Run MODELLER for a specific region"""
    
    logger.info(f"Running MODELLER for {region_name}")
    logger.info(f"  PIR: {pir_file}")
    logger.info(f"  Templates: {template_files}")
    logger.info(f"  Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the existing homology_modeller.py script
    script_dir = Path(__file__).parent
    modeller_script = script_dir / "homology_modeller.py"
    
    if not modeller_script.exists():
        logger.error(f"homology_modeller.py not found at {modeller_script}")
        return False
    
    # Prepare template arguments
    template_args = []
    for template_file in template_files:
        template_args.extend(["-t", template_file])
    
    # Build command
    cmd = [
        "python", str(modeller_script),
        "-p", pir_file,
        *template_args,
        "-o", output_dir,
        "-n", "1",  # Generate 1 model per region
        "--assess",
        "--verbose"  # Flag, not value
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run command
    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
        if result.returncode == 0:
            logger.info(f"✓ Successfully generated {region_name} model")
            return True
        else:
            logger.error(f"✗ Failed to generate {region_name} model")
            logger.error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"✗ Exception running MODELLER for {region_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Regional Hybrid Homology Modeling")
    parser.add_argument("-i", "--input-pir", required=True,
                        help="Input PIR alignment file")
    parser.add_argument("-t", "--template-dir", required=True,
                        help="Directory containing template PDB files")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory for models")
    parser.add_argument("--verbose", type=int, choices=[0,1,2], default=1,
                        help="Verbosity level")
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    logger.info("Starting Regional Hybrid Modeling")
    logger.info("=" * 50)
    
    # Define regions and their preferred templates
    regions = {
        "coverage_region": {
            "templates": ["8RJJ_0001", "8RK0_0001"],
            "description": "High coverage region (8RJJ + 8RK0)"
        },
        "conformation_region": {
            "templates": ["6MEJ_0001"],
            "description": "Alternative conformation region (6MEJ only)"
        },
        "hybrid_all": {
            "templates": ["8RJJ_0001", "8RK0_0001", "6MEJ_0001"],
            "description": "All templates combined"
        }
    }
    
    # Create base output directory
    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    
    # Process each region
    for region_name, region_config in regions.items():
        logger.info(f"\nProcessing {region_name}:")
        logger.info(f"  {region_config['description']}")
        
        # Create region-specific alignment
        region_pir = base_output / f"{region_name}_alignment.pir"
        create_region_specific_alignment(
            args.input_pir,
            region_config["templates"],
            region_pir,
            logger
        )
        
        # Prepare template files
        template_files = []
        template_dir = Path(args.template_dir)
        for template_name in region_config["templates"]:
            template_file = template_dir / f"{template_name}.pdb"
            if template_file.exists():
                template_files.append(str(template_file))
            else:
                logger.warning(f"Template file not found: {template_file}")
        
        if not template_files:
            logger.error(f"No template files found for {region_name}")
            continue
        
        # Create region output directory
        region_output = base_output / f"model_{region_name}"
        
        # Run MODELLER for this region
        success = run_modeller_for_region(
            str(region_pir),
            template_files,
            str(region_output),
            region_name,
            logger
        )
        
        if success:
            logger.info(f"✓ Completed {region_name}")
        else:
            logger.error(f"✗ Failed {region_name}")
    
    logger.info("\n" + "=" * 50)
    logger.info("Regional Hybrid Modeling Complete!")
    logger.info(f"Results saved in: {base_output}")
    
    logger.info("\nNext steps:")
    logger.info("1. Compare quality metrics between regions")
    logger.info("2. Analyze conformation differences in region 519-536")
    logger.info("3. Consider structural superposition for region comparison")

if __name__ == "__main__":
    main()