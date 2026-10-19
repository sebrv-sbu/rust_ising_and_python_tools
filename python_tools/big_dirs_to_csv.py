import argparse
from pathlib import Path
import fnmatch, glob
import pandas as pd
import numpy as np
import re

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
args=parser.parse_args();

COLS=["best_configuration", "best_cost", "best_cost_fpb", "hitting_time"]
DTYPES={"best_configuration":str, "best_cost_fpb": str}
def load_old_format_run(root_dir):
  root = Path(root_dir)
  frames = []
  for f in sorted(root.iterdir()):
    is_match = fnmatch.fnmatch(f.name, args.pattern)
    if not (f.is_file() and is_match):
      continue
    with open(f) as fh:
      first = fh.readline().rstrip("\n").split("\t")
    has_header = (first == COLS)
    df = pd.read_csv(f, sep="\t", header=0 if has_header else None, names=COLS,
                     dtype=DTYPES)
    m=re.match(r"T_(\d+)", f.name)
    if m is None: 
      raise ValueError(f"file name does not conform to old_output: {f}")

    df["T"] = float(m.group(1))
    df["max_steps"]=args.max_steps
    frames.append(df)
  return pd.concat(frames, ignore_index=True)

NEW_FORMAT_REGEXP=re.compile(r"^.+_\d+_(\d+)_(\d+(?:\.\d+)?)$")

def load_new_format_run(root_dir):
  root = Path(root_dir)
  frames = []
  for f in sorted(root.iterdir()):
    is_match = fnmatch.fnmatch(f.name, args.pattern)
    if not (f.is_file() and is_match):
      continue
    with open(f) as fh:
      first = fh.readline().rstrip("\n").split("\t")
    has_header = (first == COLS)
    df = pd.read_csv(f, sep="\t", header=0 if has_header else None, 
                     names = COLS, dtype=DTYPES)
    file_name_parsed=NEW_FORMAT_REGEXP.match(f.name)
    if file_name_parsed is None:
      raise ValueError(f"file name does not conform to new_output: {f}")
      
    df["max_steps"]=int(file_name_parsed.group(1))
    df["T"]=float(file_name_parsed.group(2))
    frames.append(df)
  return pd.concat(frames, ignore_index=True)

if args.old_format:
  if args.max_steps is None:
    parser.error("--max-steps is required with --old-format")
    exit()
  load_run = load_old_format_run
else:
  load_run = load_new_format_run

data = pd.concat([load_run(d) for d in args.target_dirs], ignore_index = True)

data["cost_exact"] = np.array([int(h, 16) for h in data["best_cost_fpb"]],
                              dtype=np.uint64).view(np.float64)
best = data["cost_exact"].min()
data["hit"] = np.isclose(data["cost_exact"], best, rtol=1e-9, atol=0)

n_configs = data.loc[data["hit"], "best_configuration"].nunique()
if n_configs > 1:
  print(f"Warning: expected 1 best configuration, found {n_configs}")

data.to_csv(args.csv, index=False)

