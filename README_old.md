# Bioinformatics Scripts Collection

A collection of useful Python scripts for computational biology, structural bioinformatics, and protein analysis workflows.

## 📋 Contents

| Script | Description | Use Case |
|--------|-------------|----------|
| `homology_modeller.py` | MODELLER homology modeling pipeline | Automated protein structure modeling |
| `pdb-download.py` | PDB structure download utility | Batch download of protein structures |
| `pdb_gap_analyzer.py` | Analyze PDB files for missing residues | Structure quality assessment |
| `pdb_sequence_extractor.py` | Extract sequences from PDB files | Sequence preparation for modeling |

## 🧬 Structural Analysis Tools

### PDB Gap Analyzer
Identifies missing residues (gaps) in protein structures, essential for understanding structural discontinuities.

**Key Features:**
- Detects missing residues in PDB structures
- Multi-chain analysis support
- Structure completeness calculation
- Gap export for alignment preparation

**Quick Start:**
```bash
python pdb_gap_analyzer.py structure.pdb
python pdb_gap_analyzer.py -c A -v structure.pdb  # Specific chain, verbose
```

### PDB Sequence Extractor
Extracts amino acid sequences from PDB files with flexible output formats and gap handling.

**Key Features:**
- FASTA and PIR format output
- Gap inclusion as dashes for missing residues
- Multi-chain and multi-file support
- Non-standard amino acid handling

**Quick Start:**
```bash
python pdb_sequence_extractor.py structure.pdb                    # FASTA output
python pdb_sequence_extractor.py --pir --gaps structure.pdb       # PIR with gaps
python pdb_sequence_extractor.py -c A --gaps structure.pdb        # Specific chain
```

## 🔄 Protein Modeling

### Homology Modeller
Comprehensive MODELLER-based homology modeling pipeline with automated workflow management.

**Key Features:**
- Multi-template homology modeling
- Automated PIR file parsing
- Quality assessment with DOPE scores
- Batch model generation
- Comprehensive logging and reporting

**Quick Start:**
```bash
python homology_modeller.py -p alignment.pir -t template.pdb
python homology_modeller.py -p alignment.pir -t template1.pdb template2.pdb -n 5 --assess
```

## 📥 Data Acquisition

### PDB Download Utility
Streamlined downloading of protein structures from the Protein Data Bank.

**Quick Start:**
```bash
python pdb-download.py 1ABC 2XYZ    # Download specific PDB IDs
```

## 🚀 Installation & Requirements

### General Requirements
- Python 3.6 or higher
- Standard Python libraries (argparse, pathlib, logging)

### Specific Requirements by Tool

**MODELLER Scripts:**
- MODELLER software package ([installation guide](https://salilab.org/modeller/))
- Proper MODELLER licensing

**PDB Analysis Tools:**
- No additional dependencies (pure Python)

### Setup
```bash
git clone <repository-url>
cd Scripts
chmod +x *.py  # Make scripts executable
```

## 📖 Usage Examples

### Complete Homology Modeling Workflow
```bash
# 1. Analyze template structures
python pdb_gap_analyzer.py template1.pdb template2.pdb

# 2. Extract template sequences with gaps
python pdb_sequence_extractor.py --pir --gaps template1.pdb template2.pdb > templates.pir

# 3. Create alignment file (manual step - combine target + templates)

# 4. Run homology modeling
python homology_modeller.py -p alignment.pir -t template1.pdb template2.pdb -n 3 --assess
```

### Structure Analysis Pipeline
```bash
# Batch analysis of multiple structures
python pdb_gap_analyzer.py *.pdb
python pdb_sequence_extractor.py *.pdb > all_sequences.fasta
```

## 📁 Output Organization

Scripts are designed to keep the repository clean:
- Log files are created in output directories (not in repo)
- Temporary files are properly handled
- `.gitignore` prevents unwanted file tracking

## 🔧 Advanced Usage

Each script includes comprehensive help:
```bash
python script_name.py --help
```

For detailed documentation on specific scripts, see their individual help output and docstrings.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements.

## 📄 License

These scripts are provided as-is for academic and research purposes.
