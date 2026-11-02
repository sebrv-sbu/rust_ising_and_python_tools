import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar, brentq
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
parser.add_argument("--chunk-rows", type=int, 
                    default=2_000_000)
parser.add_argument("--max-steps-file", type=str, default=None,
                    help="File specifiying maximum steps M for individual " +
                    "temperatures")
parser.add_argument("--n-points", type=int, default=36,
                    help="number of points")

args=parser.parse_args();
args.rust_binary = Path(args.rust_binary).expanduser()
args.ising_file = Path(args.ising_file).expanduser()

REQUIRED_COLS=["T", "max_steps", "hitting_time", "hit"]

header = pd.read_csv(args.input_csv, nrows=0).columns
missing = [c for c in REQUIRED_COLS if c not in header]
if missing:
  raise ValueError(f"{args.input_csv} is missing columns {missing};"
                   f"found {header}")

def load_max_steps(file):
  df= pd.read_csv(file, sep=r"\s+", header=0)
  return dict(zip(df["T"].astype(float), df["M"].astype(int)))

max_steps=(
  load_max_steps(args.max_steps_file)
  if args.max_steps_file is not None
  else {}
)



def merge_parts(parts):
  u = np.concatenate([p[0] for p in parts])
  c = np.concatenate([p[1] for p in parts])
  uu, inv = np.unique(u, return_inverse=True)
  return [(uu, np.bincount(inv, weights=c).astype(np.int64))]


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

def high_conv(T, n_points=36):
  """Per-step hit probability from the high-temperature series."""
  try:
    result = subprocess.run(
      [str(args.rust_binary), args.ising_file, "high-conv",
       "-T", repr(float(T))],
      capture_output=True, text=True, check=True,
    )
    line = [l for l in result.stdout.splitlines() if l.strip()][-1]
    log2_pert, sign = [s.strip() for s in line.split(",")[:2]]
    pert = 2.0 ** float(log2_pert)
    base = 2.0 ** -n_points
    p = base + pert if sign == "-" else base - pert
    return p if p > 0 else float("nan")
  except (subprocess.CalledProcessError, ValueError, IndexError) as e:
    print(f"FAILED high-conv T={T!r}: {e}")
    return float("nan")

def bin_stats(durations, hit, edges):
  th = np.sort(hit_times)
  cum_t = np.concatenate(([0.0], np.cumsum(th)))
  idx = np.searchsorted(th, edges, side="right")     # hits <= each edge
  cum_exposure = cum_t[idx] + edges * (len(th) - idx + n_cens)
  return np.diff(idx), np.diff(cum_exposure)


def bin_stats_weighted(u, c, n_total, edges):
  # Number of hits <= each edge
  idx = np.searchsorted(u, edges, side="right")

  # Cumulative number of hits
  cum_hits = np.concatenate(([0], np.cumsum(c)))

  # Cumulative sum of hitting times, weighted by frequency
  cum_t = np.concatenate(([0.0], np.cumsum(u * c)))

  # Cumulative exposure
  cum_exposure = cum_t[idx] + edges * (n_total - cum_hits[idx])

  return np.diff(idx), np.diff(cum_exposure)
  

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

def count_plateaus(g, T, u, c, n_total, n_edges=101,
                   penalty_scale=1.0, min_hits=30, min_len=2):
  if len(u) == 0:
    return 0, [], np.array([]), np.array([])
  edges = np.logspace(np.log10(u.min()), np.log10(M), n_edges)
  n_hits_binned, exposure = bin_stats_weighted(
    u, c, n_total, edges
  )
  if n_hits_binned.sum() < min_hits:
    print(f"Warning: too few hits found at temperature {T}")
    return 0, [], n_hits_binned, exposure
  penalty = penalty_scale * np.log(n_hits_binned.sum())
  cuts = segment(n_hits_binned, exposure, penalty, min_len=min_len)
  good = [
    (i, j) for i, j in cuts
    if n_hits_binned[i:j].sum() >= min_hits
  ]
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
def fit_mixture_weighted(t_u, w, n_cens, M, b_init, 
                         n_starts = 20, seed=0):
  
  k=len(b_init)
  t_u=np.asarray(t_u, dtype=float)
  w = np.asarray(w, dtype=float)

  lo = np.log(1e-2 / M)
  bounds = [(-30, 30)] * k + [(lo, 0.0)] * k

  rng = np.random.default_rng(seed)
  best = None

  for s in range(n_starts):
    logits0 = rng.normal(0, 0.5, k) if s else np.zeros(k)
    logb0 = np.log(b_init) + (
        rng.normal(0, 0.3, k) if s else 0
    )

    th0 = np.concatenate((
        logits0,
        np.clip(logb0, lo, 0.0)
    ))

    res = minimize(
        nll_and_grad,
        th0,
        args=(k, t_u, w, n_cens, M),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds
    )

    if best is None or res.fun < best.fun:
        best = res

  a = softmax(np.concatenate(([0.0], best.x[:k])))
  b = np.exp(best.x[k:])

  order = np.argsort(b)

  return a[0], a[1:][order], b[order], best


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

def dominant_rate_reached(u, c, n_total, M, b_theory,
                          a_min=0.05, delta=0.05, frac=0.2):
  h_half, h_end, eps = tail_flatness(u, c, n_total, M, frac)
  tail = h_end - h_half                           # fraction of all runs
  tol = max(delta * a_min, 3.0 / n_total)         # never below ~3 runs
  flat = tail <= tol
  b_hidden = 3.0/(n_total * a_min * (1-frac) * M)
  return {"mode_complete": bool(flat),
       "tail_fraction": tail,
       "tail_efolds": eps,
        "more_than_90": bool(h_end >= 0.9),
       "theory_testable": bool(b_theory >= b_hidden)}

def tail_flatness(u, c, n_total, M, frac=0.5):
  h_half = c[u <= frac * M].sum()
  h_end = c[u <= M].sum()
  u_half = n_total - h_half
  if u_half == 0:                              
    return h_half / n_total, h_end / n_total, 0.0
  delta = h_end - h_half
  if delta >= u_half:
      return h_half / n_total, h_end / n_total, np.inf
  return h_half / n_total, h_end / n_total, -np.log1p(-delta / u_half)

def error_bar(u, c, n_total, M, b0, level=3.841):
  h = c.sum()
  if h == 0:
    raise RuntimeError("no hits")
  n_cens = n_total - h
  S = np.dot(u, c)

  def ll(b):
    q = -np.expm1(-b * M)    # P(hit by M | eventually hits)
    if h >= n_total * q:     # best a is capped at 1
      return h * np.log(b) - b * S - n_cens * b * M
    return (
        h * np.log(h / (n_total * q)) 
        + h * np.log(b) 
        - b * S
        + n_cens * np.log(n_cens / n_total)
    )

  f = lambda x: -ll(np.exp(x))             # work in log b
  x0 = np.log(b0)
  r = minimize_scalar(f, bounds=(x0 - 6, x0 + 6), method="bounded")
  x_hat, nll_min = r.x, r.fun
  g = lambda x: f(x) - nll_min - level / 2

  def bracket(sign):
    step = 0.1
    while step <= 20:                    # stay within a factor e^20 of b_hat
      x = x_hat + sign * step
      if g(x) > 0:
        return x
      step *= 2
    raise RuntimeError("could not bracket the profile interval")

  lo = brentq(g, bracket(-1), x_hat)
  hi = brentq(g, x_hat, bracket(+1))
  return np.exp(lo), np.exp(x_hat), np.exp(hi)

def error_bar_boot(u, c, n_total, M, b_init, B=300,
                 level=0.95, seed=0, n_starts=3):
  u = np.asarray(u, dtype=float)
  c = np.asarray(c, dtype=int)

  p_hit = c.sum() / n_total
  probs = c / c.sum()

  rng = np.random.default_rng(seed)

  rates = []
  first_err = None

  for _ in range(B):

    # Number of successful runs
    h = rng.binomial(n_total, p_hit)

    if h == 0:
        continue

    # Bootstrap histogram
    w_b = rng.multinomial(h, probs)

    # Remove zero-frequency observations
    mask = w_b > 0
    t_b = u[mask]
    w_b = w_b[mask]

    try:
      _, _, exp_rates, _ = fit_mixture_weighted(
          t_b,
          w_b,
          n_cens=n_total - h,
          M=M,
          b_init=b_init,
          n_starts=n_starts
      )

      r = exp_rates[0]

      if np.isfinite(r) and r > 0:
        rates.append(r)

    except Exception as e:
      if first_err is None:
        first_err = repr(e)

  if len(rates) < 0.8 * B:
    print(
      f"bootstrap failed ({len(rates)}/{B} ok); "
      f"first error: {first_err}"
    )
    return np.nan, np.nan, 1 - len(rates) / B

  alpha = (1 - level) / 2
  lo, hi = np.quantile(rates, [alpha, 1 - alpha])

  return lo, hi, 1 - len(rates) / B
  
rows=[]
summary={}
for chunk in pd.read_csv(args.input_csv, usecols=REQUIRED_COLS, 
                         dtype={"hit":bool}, chunksize=args.chunk_rows):
  for T, g in chunk.groupby("T"):
    ms = g["max_steps"].unique()
    matching_T = [
        t for t in max_steps
        if np.isclose(T,t ,rtol=1e-8, atol=1e-12)
        ]
    if len(matching_T) > 1:
      raise ValueError(f"Ambiguous temperature match for T={T}")
    if matching_T:
      M = max_steps[matching_T[0]]
      g=g[g["max_steps"]==M]
      if g.empty:
        continue
    else: 
      if len(ms) != 1:
        raise ValueError(f"max_steps is not constant at T={T}")
      M = ms[0]
    s = summary.setdefault(T, {"n_total":0, "M":None, "parts":[]})
    s["M"] = ms[0]
    s["n_total"]+=len(g)
    t = g.loc[g["hit"], "hitting_time"].to_numpy(dtype=float)
    if np.isnan(t).any():
      raise ValueError(f"NaN hitting_time on a hit row at T={T}")
    if t.size:
      s["parts"].append(np.unique(t, return_counts=True))
      if len(s["parts"])>32:
        s["parts"] = merge_parts(s["parts"])

########
  
for T in sorted(summary):
  s=summary.pop(T)
  n_total, M=s["n_total"], s["M"]
  if s["parts"]:
    u, c = merge_parts(s["parts"])[0]
  else:
    u=np.empty(0)
    c=np.empty(0, dtype=int)
  n_hits = int(c.sum())
  n_cens = n_total - n_hits
  print(f"Temperature {T} reached")
  row = {"T": T, "n_runs":n_total, "n_hits":n_hits}
  
  min_hits=30
  min_weight=0.1
  n_plateaus, cuts, n_hits_binned, exposure = count_plateaus(
      g, T, u, c, n_total, penalty_scale=0.5
      )
  cuts = merge_similar(cuts, n_hits_binned, exposure, min_ratio=2.0)
  rates, rel_err, hits = plateau_rates(cuts, n_hits_binned, exposure)
  keep = (hits >= min_hits) & (rates > 0) & (rates < 1)
  rates = rates[keep]
  b_init=-np.log1p(-rates)
  k = len(b_init)
  one_peak_fallback = False
  a0=1
  if k == 0 and n_hits > 0:
    b_init = np.array([
      n_hits / (np.dot(u, c) + n_cens * M)
    ])
    k = 1
    one_peak_fallback = True
    #print(f"T={T}: too few hits for plateau counting, using one exponential") 
  if k == 0:
    exp_rates2=[np.nan]
    coeffs2=[np.nan]
    lo=np.nan
    hi=np.nan
    frac_failed=np.nan
  else:
    _, coeffs, exp_rates, res = fit_mixture_weighted(
        u, c, n_cens, M, b_init=b_init
        )
    keep = coeffs >= min_weight
    if not keep.any():
      keep[np.argmax(coeffs)] = True
    first = (coeffs, exp_rates)
    a0, coeffs2, exp_rates2, res2 = fit_mixture_weighted(
        u, c, n_cens, M, b_init=exp_rates[keep])
    k=len(coeffs2)
    if k == 1: 
      try:
        lo, _, hi = error_bar(u, c, n_total, M, exp_rates2[0])
      except RuntimeError:
        lo = hi = np.nan
      frac_failed=0.0
    else:
      lo, hi, frac_failed = error_bar_boot(u, c, n_total, M, exp_rates2)
  if a0 > 0.1:
    row["high_zero_weight"] = True
  else:
    row["high_zero_weight"] = False

  row[f"n_plateaus"] = k
  row["n_exponentials"] = k
  row["empirical_dominant_rate"] = exp_rates2[0]
  row["dominant_weight"] = coeffs2[0]
  row["low_temp_approx"] = low_conv(T)
  row["asymptotic_approx"] = asymptotic_match(T)
  row["high_temp_approx"] = high_conv(T, args.n_points)
  row["error_bar_lo"]=lo
  row["error_bar_hi"]=hi
  row["boot_frac_failed"] = frac_failed
  row["hit_frac"] = n_hits/n_total
  row.update(dominant_rate_reached(u, c, n_total, M, asymptotic_match(T)))
  rows.append(row)



output_data = pd.DataFrame(rows)
output_data.to_csv(args.output_csv, index=False)
