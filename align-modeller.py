from modeller import Environ, Alignment, Model, log
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Align multiple PDB templates with a target FASTA for MODELLER")
    parser.add_argument("--template_pdb", nargs="+", required=True, help="Template PDB file(s)")
    parser.add_argument("--template_code", nargs="+", required=True, help="Template code(s)")
    parser.add_argument("--target_fasta", required=True, help="Target sequence FASTA file")
    parser.add_argument("--target_code", required=True, help="Target sequence code")
    parser.add_argument("--chain", default="B", help="Chain ID to use for templates (default: B)")
    parser.add_argument("--out", default=None, help="Output alignment file (.ali)")
    args = parser.parse_args()

    log.verbose()
    env = Environ()
    env.io.atom_files_directory = ['.', './Modeller-templates']

    aln = Alignment(env)

    # Add each template model
    for pdb_file, code in zip(args.template_pdb, args.template_code):
        pdb_base = os.path.splitext(os.path.basename(pdb_file))[0]
        print(f"🧩 Adding template {pdb_base} (chain {args.chain})")
        mdl = Model(env, file=pdb_file, model_segment=(f"FIRST:{args.chain}", f"LAST:{args.chain}"))
        aln.append_model(mdl, atom_files=pdb_file, align_codes=code)


    # Add target sequence
    print(f"🎯 Adding target sequence from {args.target_fasta}")
    aln.append(file=args.target_fasta, alignment_format='FASTA', align_codes=args.target_code)

    # Align templates and target
    print("🔄 Running MODELLER align2d() ...")
    aln.align2d()

    # Output filenames
    out_ali = args.out or f"alignment_{'_'.join(args.template_code)}_{args.target_code}.ali"
    out_fasta = os.path.splitext(out_ali)[0] + ".fasta"

    # Write outputs
    aln.write(file=out_ali, alignment_format="PIR")
    aln.write(file=out_fasta, alignment_format="FASTA")

    print(f"\n✅ Alignment complete!")
    print(f"   PIR:   {out_ali}")
    print(f"   FASTA: {out_fasta}\n")

if __name__ == "__main__":
    main()
