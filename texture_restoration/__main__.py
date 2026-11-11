import argparse
from . import restore_texture

parser = argparse.ArgumentParser(description="Restore 3DGS texture using calibrated image pairs")
parser.add_argument("--ply", required=True)
parser.add_argument("--views", required=True, help="JSON manifest with rendered/original images and cameras")
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--hf-weight", type=float, default=0.01)
parser.add_argument("--background", type=float, nargs=3, default=[0, 0, 0])
args = parser.parse_args()
restore_texture(args.ply, args.views, args.output, steps=args.steps,
                lr=args.lr, hf_weight=args.hf_weight, background=args.background)
