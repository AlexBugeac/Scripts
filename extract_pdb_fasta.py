from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os, sys

aa_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

parser = PDBParser(QUIET=True)

for pdb_path in sys.argv[1:]:
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.id[0] == ' ']
            if not residues:
                continue

            seq = []
            last_num = residues[0].id[1]
            for res in residues:
                num = res.id[1]
                if num > last_num + 1:
                    seq.append('-' * (num - last_num - 1))
                seq.append(aa_map.get(res.resname, 'X'))
                last_num = num

            seq_str = ''.join(seq)
            rec = SeqRecord(Seq(seq_str),
                            id=f"{os.path.basename(pdb_path)}_chain{chain.id}",
                            description="")
            out_file = os.path.splitext(pdb_path)[0] + ".fasta"
            SeqIO.write(rec, out_file, "fasta")

            print(f"✅ {out_file}: chain {chain.id}, {len(seq_str)} residues (with gaps)")
