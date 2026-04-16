#!/usr/bin/env python3


import argparse
import mdtraj as md
import os


def main():
    parser = argparse.ArgumentParser(
        description="Downsample one or more DCDs using MDTraj stride and merge into one output DCD."
    )
    parser.add_argument(
        "-t", "--topology", required=True,
        help="Topology file (PDB or PRMTOP)"
    )
    parser.add_argument(
        "-d", "--dcd", required=True, nargs="+",
        help="Input DCD file(s), e.g. -d traj.dcd traj2.dcd traj3.dcd"
    )
    parser.add_argument(
        "-s", "--stride", type=int, default=15,
        help="Stride (keep every Nth frame, default=15)"
    )
    parser.add_argument(
        "-o", "--out",
        help="Output DCD filename (optional)"
    )

    args = parser.parse_args()

    # Output name

    if args.out:
        out_dcd = args.out
    else:
        base = os.path.splitext(os.path.basename(args.dcd[0]))[0]
        out_dcd = f"{base}_merged_stride{args.stride}.dcd"

    trajs = []
    kept_total = 0

    for dcd_path in args.dcd:
        print(f"Loading {dcd_path} (stride={args.stride})...")
        t = md.load(dcd_path, top=args.topology, stride=args.stride)
        trajs.append(t)
        kept_total += t.n_frames

    print(f"Chunks loaded : {len(trajs)}")
    print(f"Kept frames   : {kept_total}")
    print(f"Stride        : {args.stride}")

    print("Merging chunks...")
    merged = trajs[0]
    for t in trajs[1:]:
        merged = merged.join(t)

    print(f"Writing merged DCD → {out_dcd}")
    merged.save_dcd(out_dcd)

    print("Done.")


if __name__ == "__main__":
    main()
