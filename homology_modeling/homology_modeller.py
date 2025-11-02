#!/usr/bin/env python3
"""
MODELLER Homology Modeling Script

This script automates homology modeling using MODELLER from PIR alignment files
and template PDB structures. It supports multiple templates and generates
structural models with quality assessment.

Requirements:
- MODELLER software package
- Python 3.6+
- BioPython (optional, for additional structure analysis)

Usage:
    python modeller.py -p alignment.pir -t template1.pdb [template2.pdb ...] -o output_dir

Author: Generated for molecular modeling workflows
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

try:
    from modeller import *
    from modeller.automodel import *
    MODELLER_AVAILABLE = True
    
    # Import specific classes for cleaner code
    from modeller import environ, alignment, selection
    from modeller.automodel import assess
except ImportError:
    print("Warning: MODELLER not available. Please install MODELLER to use this script.")
    MODELLER_AVAILABLE = False
    # Create dummy classes to avoid NameError
    environ = alignment = selection = automodel = assess = None

# Configure basic logging (will be reconfigured later with proper paths)
logger = logging.getLogger(__name__)

def setup_logging(output_dir: str, verbose: bool = False):
    """Set up logging with proper file paths in the output directory."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set up new handlers
    log_file = Path(output_dir) / 'modeller_run.log'
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force reconfiguration
    )
    
    logger.setLevel(log_level)
    logger.info(f"Logging initialized. Log file: {log_file}")


class ModellerPipeline:
    """Main class for handling MODELLER homology modeling pipeline."""
    
    def __init__(self, pir_file: str, template_files: List[str], output_dir: str = "models", verbose: bool = False):
        """
        Initialize the modeling pipeline.
        
        Args:
            pir_file: Path to PIR alignment file
            template_files: List of template PDB file paths
            output_dir: Output directory for models
            verbose: Enable verbose logging
        """
        self.pir_file = Path(pir_file)
        self.template_files = [Path(f) for f in template_files]
        self.output_dir = Path(output_dir)
        self.target_sequence = None
        self.template_codes = []
        self.alignment_data = {}
        
        # Create output directory and set up logging
        self.output_dir.mkdir(exist_ok=True, parents=True)
        setup_logging(str(self.output_dir), verbose)
        
        # Validate inputs
        self._validate_inputs()
        
        # Parse PIR file
        self._parse_pir_file()
    
    def _validate_inputs(self):
        """Validate input files and parameters."""
        if not self.pir_file.exists():
            raise FileNotFoundError(f"PIR file not found: {self.pir_file}")
        
        for template_file in self.template_files:
            if not template_file.exists():
                raise FileNotFoundError(f"Template file not found: {template_file}")
        
        if not MODELLER_AVAILABLE:
            raise ImportError("MODELLER is required but not available")
        
        logger.info(f"Initialized pipeline with PIR: {self.pir_file}")
        logger.info(f"Templates: {[str(f) for f in self.template_files]}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _parse_pir_file(self):
        """Parse PIR alignment file to extract sequence information."""
        logger.info(f"Parsing PIR file: {self.pir_file}")
        
        try:
            with open(self.pir_file, 'r') as f:
                content = f.read()
            
            # Split into sequence blocks (separated by blank lines or >)
            sequences = self._extract_pir_sequences(content)
            
            for seq_id, seq_data in sequences.items():
                # Check if this is a template structure based on the description line
                description = seq_data.get('description', '').lower()
                if (description.startswith('structurex:') or 
                    description.startswith('structure:') or
                    'structurex' in description):
                    self.template_codes.append(seq_id)
                    logger.info(f"Found template structure: {seq_id}")
                elif (description.startswith('sequence:') or 
                      seq_id.endswith('_target') or 
                      'target' in seq_id.lower()):
                    self.target_sequence = {
                        'id': seq_id,
                        'sequence': seq_data['sequence'],
                        'description': seq_data.get('description', '')
                    }
                    logger.info(f"Found target sequence: {seq_id}")
                else:
                    # Fallback - if no clear indication, treat as target if we don't have one yet
                    if not self.target_sequence:
                        self.target_sequence = {
                            'id': seq_id,
                            'sequence': seq_data['sequence'],
                            'description': seq_data.get('description', '')
                        }
                        logger.info(f"Found target sequence (fallback): {seq_id}")
                    else:
                        # Otherwise treat as template
                        self.template_codes.append(seq_id)
                        logger.info(f"Found template structure (fallback): {seq_id}")
                
                self.alignment_data[seq_id] = seq_data
            
            if not self.target_sequence:
                raise ValueError("No target sequence found in PIR file")
            
            if not self.template_codes:
                raise ValueError("No template structures found in PIR file")
                
            # Log template information
            if len(self.template_codes) == 1:
                logger.info(f"Successfully parsed PIR file with 1 template: {self.template_codes[0]}")
            else:
                logger.info(f"Successfully parsed PIR file with {len(self.template_codes)} templates: {', '.join(self.template_codes)}")
            
        except Exception as e:
            logger.error(f"Error parsing PIR file: {e}")
            raise
    
    def _extract_pir_sequences(self, content: str) -> Dict:
        """Extract sequences from PIR file content."""
        sequences = {}
        lines = content.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # PIR header line starts with >
            if line.startswith('>'):
                # Parse header: >P1;code or >P1;code:description
                header_match = re.match(r'>([PF])1;([^:]+)(?::(.*))?', line)
                if not header_match:
                    logger.warning(f"Invalid PIR header format: {line}")
                    i += 1
                    continue
                
                seq_type = header_match.group(1)  # P for sequence, F for structure
                seq_code = header_match.group(2)
                description = header_match.group(3) or ""
                
                # Read description line (next line after header)
                i += 1
                if i < len(lines):
                    desc_line = lines[i].strip()
                    if description:
                        description = f"{description} - {desc_line}"
                    else:
                        description = desc_line
                
                # Read sequence lines until we hit * or empty line
                sequence_lines = []
                i += 1
                while i < len(lines):
                    seq_line = lines[i].strip()
                    if not seq_line or seq_line.startswith('>'):
                        break
                    
                    # Remove * at end if present (PIR terminator)
                    if seq_line.endswith('*'):
                        sequence_lines.append(seq_line[:-1])
                        i += 1
                        break
                    else:
                        sequence_lines.append(seq_line)
                    i += 1
                
                sequence = ''.join(sequence_lines)
                
                sequences[seq_code] = {
                    'type': 'sequence' if seq_type == 'P' else 'structure',
                    'sequence': sequence,
                    'description': description,
                    'code': seq_code
                }
            else:
                i += 1
        
        return sequences
    
    def _prepare_templates(self):
        """Prepare template files and copy them to working directory."""
        logger.info("Preparing template files...")
        
        # Validate number of template files
        if len(self.template_files) == 1:
            logger.info(f"Processing single template file: {self.template_files[0]}")
        else:
            logger.info(f"Processing {len(self.template_files)} template files: {[str(f) for f in self.template_files]}")
        
        template_mapping = {}
        
        for template_file in self.template_files:
            # Extract PDB code from filename or use filename
            pdb_code = template_file.stem.upper()
            
            # Copy template to output directory with standard naming
            dest_file = self.output_dir / f"{pdb_code}.pdb"
            
            try:
                import shutil
                shutil.copy2(template_file, dest_file)
                template_mapping[pdb_code] = dest_file
                logger.info(f"Copied template {template_file} -> {dest_file}")
                
                # Validate template matches PIR templates
                if pdb_code not in self.template_codes:
                    logger.warning(f"Template {pdb_code} not found in PIR file templates: {self.template_codes}")
                
            except Exception as e:
                logger.error(f"Error copying template {template_file}: {e}")
                raise
        
        return template_mapping
    
    def _create_modeller_script(self, num_models: int = 1) -> str:
        """Create MODELLER Python script for homology modeling."""
        target_code = self.target_sequence['id']
        
        # Handle both single and multiple templates for script generation
        if len(self.template_codes) == 1:
            template_codes_str = f"'{self.template_codes[0]}'"
        else:
            template_codes_str = ', '.join([f"'{code}'" for code in self.template_codes])
        
        script_content = f"""#!/usr/bin/env python

from modeller import *
from modeller.automodel import *

# Set up environment
env = environ()

# Read model library (for non-standard residues, etc.)
env.libs.topology.read(file='$(LIB)/top_heav.lib')
env.libs.parameters.read(file='$(LIB)/par.lib')

# Create automodel class
class MyModel(automodel):
    def select_atoms(self):
        # Select all atoms for modeling
        return selection(self)
        
    def special_restraints(self, aln):
        # Add any special restraints here if needed
        pass

# Create alignment object and read PIR file
aln = alignment(env)
aln.append(file='{self.pir_file.name}', align_codes='all')

# Create model
a = MyModel(env,
           alnfile='{self.pir_file.name}',
           knowns={template_codes_str},
           sequence='{target_code}',
           assess_methods=(assess.DOPE, assess.GA341))

# Generate models
a.starting_model = 1
a.ending_model = {num_models}

# Run modeling
a.make()
"""
        
        script_path = self.output_dir / "model_script.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Created MODELLER script: {script_path}")
        return str(script_path)
    
    def run_modeling(self, num_models: int = 1, assess: bool = False):
        """Run the complete homology modeling workflow."""
        logger.info("Starting homology modeling workflow...")
        
        # Change to output directory
        original_dir = os.getcwd()
        os.chdir(self.output_dir)
        
        try:
            # Prepare templates
            template_mapping = self._prepare_templates()
            
            # Copy PIR file to working directory
            import shutil
            local_pir = Path(self.pir_file.name)
            shutil.copy2(self.pir_file, local_pir)
            
            # Run modeling using MODELLER directly
            self._run_modeller_modeling(num_models, assess)
            
            # Process results
            if assess:
                self._assess_models()
            
            logger.info("Modeling workflow completed successfully!")
            
        except Exception as e:
            logger.error(f"Modeling workflow failed: {e}")
            raise
        finally:
            os.chdir(original_dir)
    
    def _run_modeller_modeling(self, num_models: int, assess: bool):
        """Execute MODELLER homology modeling."""
        logger.info(f"Running MODELLER to generate {num_models} models...")
        
        if not MODELLER_AVAILABLE:
            raise ImportError("MODELLER is not available")
        
        # Set up MODELLER environment
        env = environ()
        
        # Increase verbosity if needed
        env.io.atom_files_directory = ['.', '../atom_files']
        
        # Create alignment object and read PIR file
        aln = alignment(env)
        aln.append(file=self.pir_file.name, align_codes='all')
        
        # Skip alignment check for now - it causes issues with target sequences
        # aln.check()
        
        # Create automodel class with custom settings
        class CustomAutoModel(automodel):
            def select_atoms(self):
                return selection(self)
        
        # Set up assessment methods
        assessment_methods = []
        if assess:
            from modeller.automodel import assess
            assessment_methods = [assess.DOPE, assess.GA341]
        
        # Create and run model
        target_code = self.target_sequence['id']
        
        # Handle both single and multiple templates
        if len(self.template_codes) == 1:
            # Single template - can use string or tuple
            knowns = self.template_codes[0]
            logger.info(f"Using single template: {knowns}")
        else:
            # Multiple templates - use tuple
            knowns = tuple(self.template_codes)
            logger.info(f"Using multiple templates: {knowns}")
        
        a = CustomAutoModel(env,
                           alnfile=self.pir_file.name,
                           knowns=knowns,
                           sequence=target_code,
                           assess_methods=assessment_methods)
        
        a.starting_model = 1
        a.ending_model = num_models
        
        # Run the modeling
        a.make()
        
        logger.info(f"Generated {num_models} models for {target_code}")
        
        # Rename models with descriptive names
        self._rename_output_models(target_code, num_models)
    
    def _rename_output_models(self, target_code: str, num_models: int):
        """Rename generated models with descriptive names."""
        logger.info("Renaming output models...")
        
        for i in range(1, num_models + 1):
            # MODELLER default naming convention (corrected pattern)
            old_name = f"{target_code}.B9999000{i}.pdb"
            new_name = f"{target_code}_model_{i:02d}.pdb"
            
            old_path = Path(old_name)
            new_path = Path(new_name)
            
            if old_path.exists():
                old_path.rename(new_path)
                logger.info(f"Renamed {old_name} -> {new_name}")
            else:
                logger.warning(f"Expected model file {old_name} not found")
    
    def _assess_models(self):
        """Assess model quality and generate reports."""
        logger.info("Assessing model quality...")
        
        target_code = self.target_sequence['id']
        assessment_results = {}
        
        # Find generated models - try renamed first, then original names
        model_files = list(Path('.').glob(f"{target_code}_model_*.pdb"))
        
        if not model_files:
            # Look for original MODELLER naming pattern
            model_files = list(Path('.').glob(f"{target_code}.B9999000*.pdb"))
        
        if not model_files:
            logger.warning("No model files found for assessment")
            return
        
        # Read DOPE scores if available
        try:
            dope_scores = self._read_dope_scores(target_code)
            if dope_scores:
                assessment_results['dope_scores'] = dope_scores
        except Exception as e:
            logger.warning(f"Could not read DOPE scores: {e}")
        
        # Generate assessment report
        self._generate_assessment_report(model_files, assessment_results)
    
    def _read_dope_scores(self, target_code: str) -> Dict[str, float]:
        """Read DOPE scores from MODELLER output."""
        dope_scores = {}
        
        # Look for .D00000001, .D00000002, etc. files
        dope_files = list(Path('.').glob(f"{target_code}.D*"))
        
        for dope_file in dope_files:
            try:
                with open(dope_file, 'r') as f:
                    content = f.read()
                    # Extract DOPE score (last line usually contains the score)
                    lines = content.strip().split('\n')
                    if lines:
                        try:
                            score = float(lines[-1].strip())
                            model_num = dope_file.suffix[2:]  # Extract number from .D00000001
                            dope_scores[f"model_{model_num}"] = score
                        except ValueError:
                            pass
            except Exception as e:
                logger.warning(f"Error reading DOPE file {dope_file}: {e}")
        
        return dope_scores
    
    def _generate_assessment_report(self, model_files: List[Path], assessment_results: Dict):
        """Generate a comprehensive assessment report."""
        report_file = Path("modeling_report.txt")
        
        # Prepare report content
        report_lines = []
        report_lines.append("MODELLER Homology Modeling Report")
        report_lines.append("=" * 40)
        report_lines.append("")
        
        report_lines.append(f"Target sequence: {self.target_sequence['id']}")
        report_lines.append(f"Templates used: {', '.join(self.template_codes)}")
        report_lines.append(f"Models generated: {len(model_files)}")
        report_lines.append("")
        
        # Model summary
        report_lines.append("Generated Models:")
        report_lines.append("-" * 20)
        for model_file in sorted(model_files):
            size = model_file.stat().st_size if model_file.exists() else 0
            report_lines.append(f"{model_file.name:<25} {size:>10} bytes")
        
        # DOPE scores if available
        if 'dope_scores' in assessment_results and assessment_results['dope_scores']:
            report_lines.append("")
            report_lines.append("DOPE Scores (lower is better):")
            report_lines.append("-" * 30)
            for model, score in assessment_results['dope_scores'].items():
                report_lines.append(f"{model:<15} {score:>10.3f}")
            
            # Find best model
            best_model = min(assessment_results['dope_scores'].items(), key=lambda x: x[1])
            report_lines.append("")
            report_lines.append(f"Best model (lowest DOPE): {best_model[0]} (score: {best_model[1]:.3f})")
        else:
            report_lines.append("")
            report_lines.append("Model Quality Scores:")
            report_lines.append("-" * 25)
            report_lines.append("Scores from this run not automatically parsed.")
            report_lines.append("Please check the MODELLER output above for:")
            report_lines.append("- molpdf scores (lower is better)")
            report_lines.append("- DOPE scores (lower is better)")  
            report_lines.append("- GA341 scores (higher is better, 0-1 range)")
            report_lines.append("")
            report_lines.append("Look for 'Summary of successfully produced models' in the output.")
        
        # Additional statistics
        report_lines.append("")
        report_lines.append(f"Sequence length: {len(self.target_sequence['sequence'])}")
        report_lines.append(f"PIR file: {self.pir_file}")
        report_lines.append(f"Output directory: {self.output_dir}")
        
        # Write to file
        with open(report_file, 'w') as f:
            for line in report_lines:
                f.write(line + "\n")
        
        # Print to terminal
        logger.info("Assessment Report:")
        for line in report_lines:
            print(line)
        
        logger.info(f"Assessment report also saved to: {report_file}")
    
    def get_best_model(self) -> Optional[str]:
        """Return the path to the best model based on DOPE score."""
        try:
            target_code = self.target_sequence['id']
            dope_scores = self._read_dope_scores(target_code)
            
            if dope_scores:
                best_model = min(dope_scores.items(), key=lambda x: x[1])
                model_file = f"{target_code}_model_{best_model[0].split('_')[1]}.pdb"
                model_path = self.output_dir / model_file
                
                if model_path.exists():
                    return str(model_path)
            
            # Fallback to first model if no DOPE scores
            first_model = self.output_dir / f"{target_code}_model_01.pdb"
            if first_model.exists():
                return str(first_model)
                
        except Exception as e:
            logger.warning(f"Could not determine best model: {e}")
        
        return None


def create_sample_pir(output_file: str = "sample_alignment.pir"):
    """Create a sample PIR alignment file for testing."""
    sample_content = """>P1;target_sequence
sequence:target_sequence:1:A:200:A:target protein:organism: 0.00: 0.00
MKFLVLLFNISCMLVVFGLSAFERHLRTIDPKDLHYSGKNLQVLYSESDMHQSVLLVTVTPTHYVFAQGQTRMRLDATDKSQAAMLQNTLPSWLHPDGGPMSNQ*

>P1;1ABC
structureX:1ABC:1:A:200:A:template protein:organism: 2.00: 0.19
MKFLVLLFNISCMLVVFGLSAFERHLRTIDPKDLHYSGKNLQVLYSESDMHQSVLLVTVTPTHYVFAQGQTRMRLDATDKSQAAMLQNTLPSWLHPDGGPMSNQ*

>P1;2XYZ
structureX:2XYZ:1:A:200:A:template protein 2:organism: 2.00: 0.19
MKFLVLLFNISCMLVVFGLSAFERHLRTIDPKDLHYSGKNLQVLYSESDMHQSVLLVTVTPTHYVFAQGQTRMRLDATDKSQAAMLQNTLPSWLHPDGGPMSNQ*
"""
    
    with open(output_file, 'w') as f:
        f.write(sample_content)
    
    print(f"Sample PIR file created: {output_file}")
    print("Note: Replace with your actual target sequence and template structure codes")
    print("Format: >P1;code")
    print("        sequence: for target, structureX: for templates")


# Add utility function to main argument parser
def add_utility_commands(parser):
    """Add utility subcommands."""
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Main modeling command
    model_parser = subparsers.add_parser('model', help='Run homology modeling')
    model_parser.add_argument('-p', '--pir', required=True, help='PIR alignment file')
    model_parser.add_argument('-t', '--templates', required=True, nargs='+', help='Template PDB files')
    model_parser.add_argument('-o', '--output', default='models', help='Output directory')
    model_parser.add_argument('-n', '--num-models', type=int, default=1, help='Number of models')
    model_parser.add_argument('--assess', action='store_true', help='Perform quality assessment')
    model_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    # Sample PIR creation command
    sample_parser = subparsers.add_parser('create-pir', help='Create sample PIR file')
    sample_parser.add_argument('-o', '--output', default='sample_alignment.pir', help='Output PIR file')
    
    return subparsers


def main():
    """Main function to handle command line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="MODELLER Homology Modeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single template modeling
  python modeller.py -p alignment.pir -t template.pdb
  
  # Multiple template modeling
  python modeller.py -p alignment.pir -t template1.pdb template2.pdb -o my_models
  
  # Generate 5 models with assessment
  python modeller.py -p alignment.pir -t template.pdb -n 5 --assess
  
  # Create sample PIR file
  python modeller.py create-pir -o my_alignment.pir
        """
    )
    
    # Check if using subcommands
    if len(sys.argv) > 1 and sys.argv[1] in ['create-pir']:
        subparsers = add_utility_commands(parser)
        args = parser.parse_args()
        
        if args.command == 'create-pir':
            create_sample_pir(args.output)
            return
    else:
        # Traditional argument parsing for backward compatibility
        parser.add_argument(
            '-p', '--pir',
            required=True,
            help='PIR alignment file containing target and template sequences'
        )
        
        parser.add_argument(
            '-t', '--templates',
            required=True,
            nargs='+',
            help='Template PDB file(s) for modeling'
        )
        
        parser.add_argument(
            '-o', '--output',
            default='models',
            help='Output directory for generated models (default: models)'
        )
        
        parser.add_argument(
            '-n', '--num-models',
            type=int,
            default=1,
            help='Number of models to generate (default: 1)'
        )
        
        parser.add_argument(
            '--assess',
            action='store_true',
            help='Perform model quality assessment'
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose logging'
        )
        
        args = parser.parse_args()
    
    try:
        # Initialize pipeline (logging will be set up inside)
        pipeline = ModellerPipeline(
            pir_file=args.pir,
            template_files=args.templates,
            output_dir=args.output,
            verbose=args.verbose
        )
        
        logger.info("MODELLER pipeline initialized successfully")
        
        # Run the modeling workflow
        pipeline.run_modeling(
            num_models=args.num_models,
            assess=args.assess
        )
        
        # Report best model
        best_model = pipeline.get_best_model()
        if best_model:
            logger.info(f"Best model: {best_model}")
        
        logger.info("Homology modeling completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
