#!/usr/bin/env python3
from modeller import Environ, log, forms, features, physical
from modeller.automodel import AutoModel, autosched, refine
import argparse, sys, os

# ---------- Helpers ----------
def parse_ss_pairs(pairs):
    """Parse user-provided disulfide pairs into ('resi:chain', 'resi:chain') format."""
    norm = []
    for p in pairs:
        try:
            a, b = p.split('-')
            def normalize(s):
                if ':' not in s:
                    raise ValueError
                left, right = s.split(':', 1)
                if left.isalpha():  # A:45 -> 45:A
                    chain, resi = left, right
                else:
                    resi, chain = left, right
                return f"{resi}:{chain}"
            ra, rb = normalize(a.strip()), normalize(b.strip())
            norm.append(tuple(sorted((ra, rb))))
        except Exception:
            raise ValueError(f"Bad --ss format: '{p}'. Use A:45-A:102 or 45:A-102:A")
    return sorted(set(norm))

# ---------- Custom modeller ----------
class CombinedSSModel(AutoModel):
    def __init__(self, *args, ss_pairs=None, copy_template_ss=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._ss_pairs = ss_pairs or []
        self._copy_template_ss = copy_template_ss

    def special_patches(self, aln):
        """Handles template and custom disulfides."""
        if self._copy_template_ss:
            print("🧬 Copying disulfides from template(s)...")
            try:
                self.patch_ss_templates(aln)
                print("   ✓ Template disulfides patched.")
            except Exception as e:
                print(f"   ⚠️ Could not copy template disulfides: {e}")

        if not self._ss_pairs:
            print("ℹ️  No extra --ss pairs provided.")
            return

        print("🔧 Adding user-specified disulfides:")
        for i, (ra, rb) in enumerate(self._ss_pairs, 1):
            try:
                resA = self.residues[ra]
                resB = self.residues[rb]
                self.patch(residue_type='DISU', residues=(resA, resB))
                print(f"   {i}. DISU between {ra} ↔ {rb}")
            except Exception as e:
                print(f"⚠️ Could not add DISU {ra}–{rb}: {e}")

    def special_restraints(self, aln):
        """Add geometric restraints to encourage or discourage disulfides."""
        at = self.atoms
        try:
            s41  = at['SG:41:A']
            s46  = at['SG:46:A']
            s120 = at['SG:120:A']
        except KeyError:
            print("⚠️ Could not find one of SG atoms (41/46/120); check numbering/chain.")
            return

        # Encourage formation of 41–46 disulfide (~2.05 Å)
        self.restraints.add(
            forms.gaussian(group=physical.xy_distance,
                           feature=features.distance(s41, s46),
                           mean=2.05, stdev=0.10, weight=50.0)
        )
        print("🧲 Added restraint to favor 41–46 disulfide (~2.05 Å).")

        # Discourage proximity of 46–120 (keep SG–SG ≥ 3.5 Å)
        self.restraints.add(
            forms.lower_bound(group=physical.xy_distance,
                              feature=features.distance(s46, s120),
                              mean=3.5, stdev=0.10, weight=20.0)
        )
        print("🚫 Added restraint to prevent 46–120 disulfide (<3.5 Å).")

# ---------- Main entry ----------
def main():
    p = argparse.ArgumentParser(description="Comparative modeling with controlled disulfide patching.")
    p.add_argument("--ali", required=True, help="Alignment file (.ali)")
    p.add_argument("--knowns", nargs="+", required=True, help="Template code(s) from alignment")
    p.add_argument("--sequence", required=True, help="Target sequence code from alignment")
    p.add_argument("--num_models", type=int, default=1, help="Number of models to generate")
    p.add_argument("--atom_dir", default=".", help="Directory containing PDB templates")
    p.add_argument("--out_prefix", default=None, help="Prefix for output files")
    p.add_argument("--out_dir", default=".", help="Output directory")
    p.add_argument("--ss", action="append", default=[], help="User-specified disulfides (A:45-A:102 format)")
    p.add_argument("--no_copy_template_ss", action="store_true", help="Skip copying template disulfides")
    args = p.parse_args()

    try:
        ss_pairs = parse_ss_pairs(args.ss) if args.ss else []
    except ValueError as e:
        print(e)
        sys.exit(2)

    log.verbose()
    env = Environ()

    # Disable automatic disulfide detection by distance
    env.libs.topology.read(file='$(LIB)/top_heav.lib', patch_default=False)

    env.io.atom_files_directory = ['.', args.atom_dir]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    env.io.output_path = args.out_dir

    print("🧩 MODELLER configuration:")
    print(f"   Alignment : {args.ali}")
    print(f"   Templates : {', '.join(args.knowns)}")
    print(f"   Target    : {args.sequence}")
    print(f"   Models    : {args.num_models}")
    print(f"   Output dir: {os.path.abspath(args.out_dir)}")
    print(f"   Copy template SS: {not args.no_copy_template_ss}")
    if ss_pairs:
        print("   Extra SS  : " + "; ".join([f"{a}~{b}" for a, b in ss_pairs]))
    print()

    a = CombinedSSModel(env,
                        alnfile=args.ali,
                        knowns=args.knowns,
                        sequence=args.sequence,
                        ss_pairs=ss_pairs,
                        copy_template_ss=(not args.no_copy_template_ss))

    if args.out_prefix:
        a.outputs_prefix = args.out_prefix

    a.starting_model = 1
    a.ending_model = args.num_models

    # Improve relaxation for new disulfides
    a.library_schedule = autosched.slow
    a.max_var_iterations = 300
    a.md_level = refine.slow

    print("🚀 Starting comparative modeling with disulfide control...\n")
    a.make()
    print("\n✅ Finished. Models saved to:", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()
