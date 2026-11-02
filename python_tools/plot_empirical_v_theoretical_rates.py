import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter



parser = argparse.ArgumentParser(
    description = "Reads processed csv and outputs a nice graph")
parser.add_argument("input_csv", help="input csv", metavar="input-csv")
parser.add_argument("--output-file", default="emp_v_rate.pdf", 
                    help="output file name")
parser.add_argument("--x-min", type=float, default=1e-2)
parser.add_argument("--y-min", type=float, default=None)

args=parser.parse_args();

REQUIRED_COLS=["T","n_runs","n_hits","n_plateaus","n_exponentials",
               "empirical_dominant_rate","dominant_weight","low_temp_approx",
               "asymptotic_approx","high_temp_approx","mode_complete",
               "tail_fraction","tail_efolds","theory_testable","hit_frac",
               "error_bar_lo", "error_bar_hi", "high_zero_weight"]

data=pd.read_csv(args.input_csv)

missing = [c for c in REQUIRED_COLS if c not in data.columns]
if missing:
  raise SystemExit(f"CSV is missing columns {missing}; regenerate it.")

HIT_THRESH = 0.75

rate=data["empirical_dominant_rate"]
lo_err = np.maximum(rate-data["error_bar_lo"], 0)
hi_err = np.maximum(data["error_bar_hi"]-rate, 0)
final_peak_not_reached = data["high_zero_weight"].astype(bool)
reached = data["hit_frac"] >= HIT_THRESH

fig, ax = plt.subplots(figsize=(12, 8))
for peak_reached in (True, False):
  for is_reached, color in ((True, "k"), (False, "C1")):
    m = (peak_reached != final_peak_not_reached) & (reached == is_reached)
    ax.errorbar(data.loc[m, "T"], rate[m],
                 yerr=[lo_err[m], hi_err[m]],
                 fmt="o", ms=6, color=color, capsize=2,
                 mfc=color if peak_reached else "none")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_ylabel("rate")

l2, = ax.plot(data["T"], data["high_temp_approx"], "--", color="C3")
l3, = ax.plot(data["T"], data["low_temp_approx"], ":", color="C4")
pct = HIT_THRESH *100
handles = [
    Line2D([], [], marker="o", ls="", color="k", label=rf"$\geq${pct:g}% runs hit"),
    Line2D([], [], marker="o", ls="", color="C1", label=rf"<{pct:g}% runs hit"),
    Line2D([], [], marker="o", ls="", color="gray", label="slow peak hit"),
    Line2D([], [], marker="o", ls="", color="gray", mfc="none",
           label="slow peak not hit"),
    l2, l3,
    ]
labels = [h.get_label() for h in handles]
labels[4:] = ["high temperature perturbation series",
              "low temperature perturbation series"]
ax.legend(handles, labels)
if args.y_min is None:
  y_bot = min(rate)
else:
  y_bot = args.y_min
ax.set_ylim(top=1, bottom=y_bot)
ax.set_xlim(left=args.x_min)
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2,10), numticks=200))
ax.xaxis.set_minor_formatter(NullFormatter())   # ticks but no labels
ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=20))
fig.tight_layout()
fig.savefig(args.output_file, dpi=200)
