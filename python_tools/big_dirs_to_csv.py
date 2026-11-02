import argparse
from pathlib import Path
import fnmatch, glob
import pandas as pd
import numpy as np
import re
import warnings

parser = argparse.ArgumentParser(description=("Move all ising files for one "
                                              "run type into a csv"))

parser.add_argument("target_dirs", nargs="+", help=("Which directories to scan"
    " e.g. dir1 dir2 ... "), metavar="target-dirs")
parser.add_argument("--old-format", action="store_true")
parser.add_argument("--max-steps", type=int, default=None,
                    help = ("steps each run was allowed - will not override "
                    "the value parsed from file name if using new format"))
parser.add_argument("--csv", default="concat.csv", help="output csv name");
parser.add_argument("--pattern", default="*.txt", help=("regular expression "
                    "to find files in the directory"))
parser.add_argument("--chunk-rows", type=int, default=1_000_000,
                    help="rows held in memory at once")
parser.add_argument("--best-conf", type=str, default=None,
                    help="known best cost; skips the first pass")
args=parser.parse_args();

COLS=["best_configuration", "best_cost", "best_cost_fpb", "hitting_time"]
DTYPES={"best_configuration":str, "best_cost_fpb": str}
OLD_NAME_RE = re.compile(r"^T_([0-9.]+(?:[eE][+-]?\d+)?)_")
NEW_NAME_RE = re.compile(
    r"^.+?_(\d+)_(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$"
)
def iter_files():
  for d in args.target_dirs:
    for f in sorted(Path(d).iterdir()):
      is_match = fnmatch.fnmatch(f.name, args.pattern)
      if not (f.is_file() and is_match):
        continue
      if args.old_format:
        m = OLD_NAME_RE.match(f.name)
        if m is None:
          raise ValueError(f"file name does not conform to old output: {f}")
        yield f, float(m.group(1)), args.max_steps
      else:
        m = NEW_NAME_RE.match(f.name)
        if m is None:
          warnings.warn("file name does not conform to new_output, "+
                        f"skipping: {f}")
          continue
        yield f, float(m.group(2)), int(m.group(1))

def read_chunks(f, usecols=None):
  with open(f) as fh:
    first=fh.readline().rstrip("\n").split("\t")
  return pd.read_csv(f, sep="\t", header=0 if first==COLS else None,
                     names=COLS, dtype=DTYPES, usecols=usecols, 
                     chunksize=args.chunk_rows)

def hex_to_float(s):
  bits = np.fromiter((int(h, 16) for h in s), dtype=np.uint64, count=len(s))
  return bits.view(np.float64)


if args.best_conf is not None:
  best_configuration = args.best_conf
else:
  best = np.inf
  best_configuration = None
  for f, T, ms in iter_files():
    for chunk in read_chunks(f, usecols=["best_cost_fpb", 
                                         "best_configuration"]):
      if chunk.empty: 
        warnings.warn(f"no data rows in {f}")
        continue
      costs=hex_to_float(chunk["best_cost_fpb"])
      if best > costs.min():
        i = int(costs.argmin())
        best_configuration = chunk["best_configuration"].iloc[i]
        best = costs.min()

  if best_configuration is None:
    parser.error("no files matched; cannot determine best configuration")

print(f"best configuration: {best_configuration}")

out_path = Path(args.csv)
tmp = out_path.with_name(out_path.name + ".tmp")
configs, first_write, n_rows, n_hits = set(), True, 0, 0

for f, T, ms in iter_files():
  for chunk in read_chunks(f):
    hit = (chunk["best_configuration"] == best_configuration).to_numpy()
    chunk["T"] = T
    chunk["max_steps"] = ms
    chunk["hit"] = hit
    out = chunk[["T", "max_steps", "hitting_time", "hit"]]
    out.to_csv(tmp, mode="w" if first_write else "a",
               header=first_write, index=False)
    first_write = False
    n_rows+= len(chunk)
    n_hits +=(hit.sum())

if first_write:
  parser.error("no files matched; nothing written")
tmp.replace(out_path) 
print(f"{n_rows} runs, {n_hits} hits")

