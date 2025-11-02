# Protein Modeling Scripts

A collection of Python scripts for protein structure analysis, homology modeling, and molecular dynamics simulations.

## 📁 Directory Structure

### 🧬 `homology_modeling/`
Scripts for protein homology modeling using MODELLER
- `homology_modeller.py` - Main homology modeling script with assessment
- `hybrid_homology_modeller.py` - Multi-template homology modeling
- `mutation_modeller.py` - Protein mutation modeling
- `regional_hybrid_modeller.py` - Region-specific hybrid modeling
- `simple_modeller_disulfides.py` - Basic modeling with disulfide bonds
- `modeller_with_disulfides.py` - Advanced disulfide constraint modeling
- `model_comparison.py` - Compare and assess protein models

### 🔍 `structure_analysis/`
Tools for protein structure analysis and modification
- `structure_analyzer.py` - General protein structure analysis
- `analyze_cys_distances.py` - Analyze cysteine distances for disulfide bonds
- `add_disulfide_bonds.py` - Add disulfide bond annotations to PDB files
- `add_e2_disulfides.py` - Add E2-specific disulfide bonds
- `force_add_disulfides.py` - Force add disulfide bonds regardless of distance

### 📊 `pdb_processing/`
PDB file downloading, processing, and extraction tools
- `pdb-download.py` - Download PDB files from RCSB
- `pdb_sequence_extractor.py` - Extract sequences from PDB files
- `pdb_gap_analyzer.py` - Analyze gaps and missing residues in PDB structures
- `extract_e2_chains.py` - Extract specific chains from PDB files (E2 protein focus)
- `demo_pdb_extraction.py` - Demo script for PDB chain extraction
- `forced_substitution_aligner.py` - Alignment with forced substitutions

### 🏃‍♂️ `md_simulation/`
Molecular dynamics simulation scripts
- `simulate.folding.py` - Protein folding simulations
- `simulate.equilibrium.py` - Equilibrium MD simulations

### 📂 `temp_files/`
Temporary files, model outputs, and intermediate results
- Generated PDB models
- MODELLER output files
- Alignment files (PIR format)
- Assessment results

## 🚀 Quick Start

### Requirements
```bash
# Install dependencies
conda install -c conda-forge biopython
conda install -c salilab modeller
pip install requests numpy
```

### Common Workflows

#### 1. Basic Homology Modeling
```bash
cd homology_modeling/
python homology_modeller.py -p alignment.pir -t template.pdb -o output_dir
```

#### 2. Add Disulfide Bonds to Models
```bash
cd structure_analysis/
python add_disulfide_bonds.py input_model.pdb output_with_disulfides.pdb
```

#### 3. Download and Process PDB Files
```bash
cd pdb_processing/
python pdb-download.py 1ABC
python extract_e2_chains.py 1ABC.pdb
```

#### 4. Analyze Protein Structures
```bash
cd structure_analysis/
python structure_analyzer.py protein.pdb
python analyze_cys_distances.py protein.pdb
```

## 📝 Usage Examples

### Multi-template Homology Modeling
```bash
python homology_modeling/hybrid_homology_modeller.py \
  --alignment multi_template.pir \
  --templates template1.pdb template2.pdb \
  --output models/ \
  --num_models 5
```

### E2 Protein Modeling Pipeline
```bash
# 1. Extract clean E2 chains
python pdb_processing/extract_e2_chains.py 8RJJ.pdb

# 2. Run homology modeling
python homology_modeling/simple_modeller_disulfides.py

# 3. Add disulfide bonds
python structure_analysis/add_e2_disulfides.py E2_model.pdb E2_final.pdb
```

## 🔧 Script Features

### Homology Modeling
- ✅ Single and multi-template modeling
- ✅ Model quality assessment (DOPE, GA341, molpdf)
- ✅ Disulfide bond constraints
- ✅ Mutation modeling
- ✅ Loop refinement

### Structure Analysis
- ✅ Disulfide bond detection and analysis
- ✅ Structure quality metrics
- ✅ Distance calculations
- ✅ Chain extraction and cleaning

### PDB Processing
- ✅ Automated PDB downloading
- ✅ Sequence extraction
- ✅ Gap analysis
- ✅ Chain separation
- ✅ Structure cleaning

## 📚 Dependencies

- Python 3.7+
- BioPython
- MODELLER 10.7+
- NumPy
- Requests
- Math (built-in)

## 🤝 Contributing

1. Keep scripts focused and well-documented
2. Follow the directory structure
3. Include usage examples in docstrings
4. Test scripts before committing

## 📄 License

Academic and research use. Please cite appropriate software packages when publishing results.

---

*Organized structure for protein modeling workflows*