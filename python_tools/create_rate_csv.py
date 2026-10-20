import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

parser = argparse.ArgumentParser(
    description = ("Does some MLE stuff to tack on approximate dominant "
                   "convergence rate from empirical runs and theoretical "
                   "convergence rate. Should be used before making plots.")
    )
parser.add_argument("ising_file", help="Ising file specification", 
                    metavar="ising-file")
parser.add_argument("--input-csv", default="concat.csv", help="input csv name")
parser.add_argument("--output-csv", default="rates.csv", 
                    help="output csv name")
parser.add_argument("--rust-binary", default="~/rust_ising_and_python_tools/rust_ising/target/release/rust_ising", help=("rust binary for computing "
                   "theoretical convergence values"))

args=parser.parse_args();
args.rust_binary = Path(args.rust_binary).expanduser()
args.ising_file = Path(args.ising_file).expanduser()

REQUIRED_COLS=["best_configuration", "best_cost", "best_cost_fpb", 
               "hitting_time", "T", "max_steps", "hit"]
DTYPES={"best_configuration":str, "best_cost_fpb": str}


data=pd.read_csv(args.input_csv, dtype=DTYPES)

missing = [c for c in REQUIRED_COLS if c not in data.columns]
if missing:
  raise ValueError(f"{args.input_csv} is missing columns {missing};"
                   f"found {list(data.columns)}")

def asymptotic_match(T):
  try:
    result = subprocess.run(
      [str(args.rust_binary), args.ising_file, "asymptotic-match", "-T", f"{T:.1g}"],
      capture_output=True, text=True, check=True,
    )
    line = [l for l in result.stdout.splitlines() if l.strip()][-1]
    return 1.0-float(line.split(",")[0])
  except (subprocess.CalledProcessError, ValueError, IndexError) as e:
    print(f"FAILED asymptotic-match T={T:.1g}: {e}")
    return float("nan")

def low_conv(T):
  try:
    result = subprocess.run(
        [str(args.rust_binary), args.ising_file, "low-conv", "-T", f"{T:.1g}"],
        capture_output=True, text=True, check=True,
        )
    line = [l for l in result.stdout.splitlines() if l.strip()][-1]
    return float(line.split(",")[0])
  except (subprocess.CalledProcessError, ValueError, IndexError) as e:
    print(f"FAILED low-conv T={T:.1g}: {e}")
    return float("nan")

LN2 = np.log(2.0)

def high_conv(T):
  try:
    result = subprocess.run(
      [str(args.rust_binary), args.ising_file, "high-conv", "-T", f"{T:.1g}"],
      capture_output=True, text=True, check=True,
    )
    line = [l for l in result.stdout.splitlines() if l.strip()][-1]
    x = float(line.split(",")[0])        # log2 of the value
    return np.exp(x * LN2)            # = 1 - 2**x, accurate for small |x|
  except (subprocess.CalledProcessError, ValueError, IndexError) as e:
    print(f"FAILED high-conv T={T:.1g}: {e}")
    return float("nan")
  
def bin_stats(durations, hit, edges):
  t = np.sort(durations)
  cum_t = np.concatenate(([0.0], np.cumsum(t)))
  survived = np.searchsorted(t, edges, side="right")
  cum_exposure = cum_t[survived] + edges * (len(t)-survived)
  exposure = np.diff(cum_exposure)
  th = np.sort(durations[hit])
  n_hits_binned = np.diff(np.searchsorted(th, edges, side="right"))
  return n_hits_binned, exposure

def seg_log_likelihood(n_hits,exposure):
  return 0.0 if n_hits == 0 else n_hits * np.log(n_hits / exposure) - n_hits

def segment(n_hits_binned, exposure, penalty, min_len=2):
  n_bins = len(n_hits_binned)
  cum_hits = np.concatenate(([0],np.cumsum(n_hits_binned)))
  cum_exposure = np.concatenate(([0], np.cumsum(exposure)))
  best = np.full(n_bins + 1, -np.inf)
  best[0] = 0.0
  last_cut = np.zeros(n_bins + 1, dtype=int)
  for j in range(min_len, n_bins+1):
    for i in range(0, j-min_len+1):
      if best[i] == -np.inf: continue
      seg_exposure = cum_exposure[j] - cum_exposure[i]
      seg_hits = cum_hits[j] - cum_hits[i]
      if seg_exposure <= 0 : continue
      score = best[i] + seg_log_likelihood(seg_hits, seg_exposure) - penalty
      if score > best[j]:
        best[j], last_cut[j] = score, i
  cuts, j = [], n_bins
  while j > 0: 
    cuts.append((last_cut[j], j))
    j = last_cut[j]
  return cuts[::-1]

def count_plateaus(g, T, hit_times, n_edges=101, penalty_scale=1.0, min_hits=30,
                   min_len=2):
  max_steps = g["max_steps"].iloc[0]
  hit = g["hit"].to_numpy(dtype=bool)
  durations = np.where(hit, g["hitting_time"], max_steps).astype(float)
  edges = np.logspace(np.log10(hit_times.min()), np.log10(max_steps), n_edges)
  n_hits_binned, exposure = bin_stats(durations, hit, edges)
  if n_hits_binned.sum() < min_hits:
    print("Warning: too few hits found at temperature " + str(T) + 
          " defaulting to one peak")
    return 0, [], n_hits_binned, exposure
  penalty = penalty_scale * np.log(n_hits_binned.sum())
  cuts = segment(n_hits_binned, exposure, penalty, min_len=min_len)
  good = [(i, j) for i, j in cuts if n_hits_binned[i:j].sum() >= min_hits]
  return len(good), cuts, n_hits_binned, exposure

def merge_similar(cuts, n_hits_binned, exposure, min_ratio=2.0):
  segs = [list(c) for c in cuts]
  while len(segs) > 1:
    r = [n_hits_binned[i:j].sum() / exposure[i:j].sum() for i, j in segs]
    ratios = [max(r[m], r[m+1]) / min(r[m], r[m+1]) if min(r[m], r[m+1]) > 0
            else np.inf for m in range(len(segs) - 1)]
    m = int(np.argmin(ratios))
    if ratios[m] >= min_ratio:
      break
    segs[m:m+2] = [[segs[m][0], segs[m+1][1]]]
  return [tuple(s) for s in segs]

rows = []

def fit_mixture(hit_times, n_cens, M, b_init, n_starts=20, seed=0):
  k = len(b_init)
  t_u, w = np.unique(hit_times, return_counts=True)
  t_u = t_u.astype(float)
  lo = np.log(1e-2 / M)                      # rates below ~0.01/M belong to the never-hit class
  bounds = [(-30, 30)] * k + [(lo, 0.0)] * k
  rng = np.random.default_rng(seed)
  best = None
  for s in range(n_starts):
    logits0 = rng.normal(0, 0.5, k) if s else np.zeros(k)
    logb0 = np.log(b_init) + (rng.normal(0, 0.3, k) if s else 0)
    th0 = np.concatenate((logits0, np.clip(logb0, lo, 0.0)))
    res = minimize(nll_and_grad, th0, args=(k, t_u, w, n_cens, M),
                   jac=True, method="L-BFGS-B", bounds=bounds)
    if best is None or res.fun < best.fun:
      best = res
  a = softmax(np.concatenate(([0.0], best.x[:k])))   # a[0] = never-hit class
  b = np.exp(best.x[k:])
  order = np.argsort(b)
  return a[0], a[1:][order], b[order], best

def unpack(theta, k):
  log_a = log_softmax(np.concatenate(([0.0], theta[:k-1])))
  b = np.exp(theta[k-1:])
  return log_a, b

def nll_and_grad(theta, k, t, w, n_cens, M):
  z = np.concatenate(([0.0], theta[:k]))          # k+1 logits; class 0 never hits
  zmax = z.max()
  log_a = z - (zmax + np.log(np.exp(z - zmax).sum()))
  a = np.exp(log_a)                               # a[0] never-hit, a[1:] exponentials
  b = np.exp(theta[k:])

  # hits: only the k exponential classes can produce them
  s = log_a[1:, None] - np.outer(b, t) + np.log(np.expm1(b))[:, None]
  smax = s.max(axis=0)
  e = np.exp(s - smax)
  denom = e.sum(axis=0)
  r = e / denom
  ll = (w * (smax + np.log(denom))).sum()

  # censored: all k+1 classes can produce them
  u = np.concatenate(([log_a[0]], log_a[1:] - b * M))
  umax = u.max()
  eu = np.exp(u - umax)
  su = eu.sum()
  q = eu / su
  ll += n_cens * (umax + np.log(su))

  wr = r * w
  H = wr.sum(axis=1)
  G = n_cens * q
  G[1:] += H                                      # expected count per class
  N = w.sum() + n_cens
  g_z = (G - a * N)[1:]                           # class 0 logit is fixed
  g_b = -(wr * t).sum(axis=1) + H / (-np.expm1(-b)) - n_cens * q[1:] * M
  return -ll, -np.concatenate((g_z, g_b * b))

def plateau_rates(cuts, n_hits_binned, exposure):
  rates, rel_err, hits = [], [], []
  for i, j in cuts:
    D = n_hits_binned[i:j].sum()
    E = exposure[i:j].sum()
    rates.append(D/E)
    rel_err.append(1/np.sqrt(D) if D>0 else np.inf)
    hits.append(D)
  return np.array(rates), np.array(rel_err), np.array(hits)

def dominant_flags(a0, coeffs, rates, M, min_weight=0.1, efold_min=3.0):
    a_slow, b_slow = coeffs[0], rates[0]           # sorted ascending
    efolds = b_slow * M
    return {
      "slowest_observed_rate": b_slow,
      "slowest_weight": a_slow,
      "slowest_efolds": efolds,
      "never_hit_weight": a0,
      "dominant_rate_seen": bool(efolds >= efold_min and a_slow >= min_weight),
    }

def dominant_rate_reached(hit_times, n_total, M, b_theory,
                          a_min=0.05, delta=0.05, frac=0.2):
  h_half, h_end, eps = tail_flatness(hit_times, n_total, M, frac)
  tail = h_end - h_half                           # fraction of all runs
  tol = max(delta * a_min, 3.0 / n_total)         # never below ~3 runs
  flat = tail <= tol
  b_hidden = 3.0/(n_total * a_min * (1-frac) * M)
  return {"mode_complete": bool(flat),
       "tail_fraction": tail,
       "tail_efolds": eps,
        "more_than_90": bool(h_end >= 0.9),
       "theory_testable": bool(b_theory >= b_hidden)}

def tail_flatness(hit_times, n_total, M, frac=0.5):
    h_half = np.count_nonzero(hit_times <= frac * M)
    h_end  = np.count_nonzero(hit_times <= M)
    u_half = n_total - h_half
    if u_half == 0:                              
      return h_half / n_total, h_end / n_total, 0.0
    delta = h_end - h_half
    if delta >= u_half:
        return h_half / n_total, h_end / n_total, np.inf
    return h_half / n_total, h_end / n_total, -np.log1p(-delta / u_half)

def error_bar_boot(hit_times, n_total, M, b_init, B=300,
                   level=0.95, seed=0, n_starts=3):
  def fit_dominant_rate(t_b):
    n_cens = n_total - len(t_b)
    _, _, exp_rates, _ = fit_mixture(t_b, n_cens, M, b_init=b_init, 
                                     n_starts=n_starts)
    return exp_rates[0]

  rng = np.random.default_rng(seed)
  t = np.asarray(hit_times, dtype=float)
  p_hit = t.size / n_total
  rates = []
  first_err = None
  for _ in range(B):
    h = rng.binomial(n_total, p_hit)           # how many runs hit
    t_b = rng.choice(t, size=h, replace=True)  # their hit times
    try:
      r = fit_dominant_rate(t_b)
    except Exception as e:
      if first_err is None:
        first_err = repr(e)
      continue

    if np.isfinite(r) and r > 0:
      rates.append(r)
  if len(rates) < 0.8 * B:
    print(f"bootstrap failed ({len(rates)}/{B} ok); first error: {first_err}")
    return np.nan, np.nan, 1 - len(rates) / B
  alpha = (1 - level) / 2
  lo, hi = np.quantile(rates, [alpha, 1 - alpha])
  return lo, hi, 1 - len(rates) / B

rows=[]
for T, g in data.groupby("T"):
  n_total=len(g)
  print("Temperature "+str(T)+" reached")
  row = {"T":T, "n_runs": len(g), "n_hits": int(g["hit"].sum())}
  hit = g["hit"].to_numpy(dtype=bool)
  hit_times = g.loc[hit, "hitting_time"].to_numpy(dtype=float)
  n_cens=int((~hit).sum())
  M=g["max_steps"].iloc[0]

  min_hits=30
  min_weight=0.1
  n_plateaus, cuts, n_hits_binned, exposure = count_plateaus(
      g, T, hit_times, penalty_scale=0.5
      )
  cuts = merge_similar(cuts, n_hits_binned, exposure, min_ratio=2.0)
  rates, rel_err, hits = plateau_rates(cuts, n_hits_binned, exposure)
  keep = (hits >= min_hits) & (rates > 0) & (rates < 1)
  rates = rates[keep]
  b_init=-np.log1p(-rates)
  k = len(b_init)
  if k == 0:
    exp_rates2=[0]
    coeffs2=[0]
    lo=np.nan
    hi=np.nan
  else:
    _, coeffs, exp_rates, res = fit_mixture(hit_times, n_cens, M, b_init=b_init)
    keep = coeffs >= min_weight
    if not keep.any():
      keep[np.argmax(coeffs)] = True
    first = (coeffs, exp_rates)
    a0, coeffs2, exp_rates2, res2 = fit_mixture(hit_times, n_cens, M,
                                            b_init=exp_rates[keep])
    k=len(coeffs2)
    lo, hi, frac_failed = error_bar_boot(hit_times, n_total, M, exp_rates2)
  row[f"n_plateaus"] = k
  row["n_exponentials"] = k
  row["empirical_dominant_rate"] = exp_rates2[0]
  row["dominant_weight"] = coeffs2[0]
  row["low_temp_approx"] = low_conv(T)
  row["asymptotic_approx"] = asymptotic_match(T)
  row["high_temp_approx"] = high_conv(T)
  row["error_bar_lo"]=lo
  row["error_bar_hi"]=hi
  row["boot_frac_failed"] = frac_failed
  row["hit_frac"] = len(hit_times)/n_total
  row.update(dominant_rate_reached(hit_times, n_total, M, asymptotic_match(T)))
  rows.append(row)



output_data = pd.DataFrame(rows)
output_data.to_csv(args.output_csv, index=False)
