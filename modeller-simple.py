from modeller import Environ, log
from modeller.automodel import AutoModel
import argparse
import os

# --- Custom class to automatically patch disulfides from templates ---
class MyModel(AutoModel):
    def special_patches(self, aln):
        """Automatically copy disulfide bonds from template(s)"""
        self.patch_ss_templates(aln)
        print("🧬 Applied disulfide patches from templates.")

def main():
    parser = argparse.ArgumentParser(description="Run MODELLER AutoModel with multiple templates + SS patching")
    parser.add_argument("--ali", required=True, help="Alignment file (.ali)")
    parser.add_argument("--knowns", nargs="+", required=True, help="Template code(s)")
    parser.add_argument("--sequence", required=True, help="Target sequence code (from .ali)")
    parser.add_argument("--num_models", type=int, default=1, help="Number of models to generate")
    parser.add_argument("--atom_dir", default=".", help="Directory containing PDB templates")
    args = parser.parse_args()

    log.verbose()
    env = Environ()
    env.io.atom_files_directory = ['.', args.atom_dir]

    print("🧩 MODELLER configuration:")
    print(f"   Alignment file : {args.ali}")
    print(f"   Templates      : {', '.join(args.knowns)}")
    print(f"   Target         : {args.sequence}")
    print(f"   Models to run  : {args.num_models}\n")

    # --- Run AutoModel with disulfide patching ---
    a = MyModel(env,
                alnfile=args.ali,
                knowns=args.knowns,
                sequence=args.sequence)
    a.starting_model = 1
    a.ending_model = args.num_models

    print("🚀 Starting comparative modeling with disulfide patching...\n")
    a.make()
    print("\n✅ Modeling finished successfully with disulfide patches applied!")

if __name__ == "__main__":
    main()
