#!/usr/bin/env python3
"""
Generate an Ising model input file.

Supports:
  - Ferromagnetic model (J > 0, default)
  - Edwards-Anderson spin glass (random J from normal or uniform distribution)
  - Planting an arbitrary ground state via gauge transformation
  - Requiring at least N local minima (enumerative check, bails if too large)

The Hamiltonian is H = -sum J_ij sigma_i sigma_j - sum h_i sigma_i.
Spins: +1 (true/up) or -1 (false/down).

Strategy:
 1. Build the system so that all-zeros (all spin-down / -1) is a local minimum.
 2. Enumerate and count local minima.
 3. If a different ground state is requested, apply a gauge transformation:
       J_ij -> J_ij * sigma*_i * sigma*_j,   h_i -> h_i * sigma*_i
    which maps all-zeros -> -sigma* (the complement). We apply the
    transform to the complement so that the user's target becomes the image.

Usage:
    python3 generate_ising.py
    python3 generate_ising.py --model edwards-anderson --j-dist normal --j-mean 0 --j-std 1.5
    python3 generate_ising.py --ground-state 01001100 --min-local-minima 3
    python3 generate_ising.py --model edwars-anderson --ground-state 0101 --min-local-minima 5 --max-attempts 500
    python3 generate_ising.py --dim 3 --sizes 4 4 4 --seed 42
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constnts
# ---------------------------------------------------------------------------

MAX_ENUMERATION_SPINS = 24  # 2^24 ~ 16Mconfigs; feasible upper bound

# ---------------------------------------------------------------------------
# Helprs
# ---------------------------------------------------------------------------


def _parse_spin_bool(c: str) -> bool:
    """'1' -> True (+1), '0' -> False (-1)."""
    if c == "1":
        return True
    if c == "0":
        return False
    raise ValueError(f"Invalid spin chrcter: {c!r} (expected '0' or '1')")


def _spin_to_int(spin: bool) -> int:
    """True -> +1, False -> -1."""
    return 1 if spin else -1


def _int_to_spin(v: int) -> bool:
    """+1 -> True, -1 -> False."""
    return v == 1


def _read_ground_state(raw: str, n_points: int) -> list[bool]:
    """Parse a ground-state string: either a bitstring or path to a file
    containing one (whitespace stripped)."""
    path = Path(raw)
    if path.is_file():
        raw = path.read_text().strip()
    raw = "".join(raw.split())  # strip all whitespace
    if len(raw) != n_points:
        raise ValueError(
            f"Ground state length {len(raw)} != number of pins {n_points}"
        )
    return [_parse_spin_bool(c) for c in raw]


def _build_topology(
    dim: int, sizes: list[int]
) -> tuple[int, list[list[int]], list[tuple[int, int]]]:
    """Return (n_points, neighbours_per_node, undirected_edges).

    neighbours_per_node[node] = [nbr0, nbr1, ...] in the same order the
    Rust reader expects (each dimension: +dir then -dir).
    undirected_edges = [(a, b), ...] with a < b, in outputorder.
   """
    n_points = 1
    for s in sizes:
        n_points *= s

    def to_coord(node: int) -> list[int]:
        coord = []
        n = node
        for s in sizes:
            coord.append(n % s)
            n //= s
        return coord

    def from_coord(coord: list[int]) -> int:
        base = 1
        node = 0
        for pos, s in zip(coord, sizes):
            node += pos * base
            base *= s
        return node

    neighbours: list[list[int]] = []
    edges: list[tuple[int, int]] = []

    for node in range(n_points):
        nbrs = []
        coord = to_coord(node)
        for i, s in enumerate(sizes):
            if s > 1:
                orig = coord[i]
                coord[i] = (orig + 1) % s
                nbrs.append(from_coord(coord))
                coord[i] = (orig + s - 1) % s
                nbrs.append(from_coord(coord))
                coord[i] = orig
            else:
                nbrs.append(node)
                nbrs.append(node)
        neighbours.append(nbrs)
        for nbr in nbrs:
            if nbr > node:
                edges.append((node, nbr))

    return n_points, neighbours, edges


def _generate_edge_couplings(
    edges: list[tuple[int, int]],
    model: str,
    j_dist: str,
    j_mean: float,
    j_std: float,
    j_min: float,
    j_max: float,
    rng: random.Random,
) -> dict[tuple[int, int], float]:
    """Return {(a, b): J_ab} for all undirected edges."""
    edge_j: dict[tuple[int, int], float] = {}
    for a, b in edges:
        if model == "ferro":
            v = rng.uniform(j_min, j_max)
        elif j_dist == "normal":
            v = rng.gauss(j_mean, j_std)
        else:  # uniform
            v = rng.uniform(j_min, j_max)
        edge_j[(a, b)] = v
    return edge_j


def _compute_node_neighbour_j(
    n_points: int,
    neighbours: list[list[int]],
    edge_j: dict[tuple[int, int], float],
) -> list[list[tuple[int, float]]]:
    """Return node_nbrs[node] = [(nbr, J), ...]."""
    node_nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n_points)]
    for node in range(n_points):
        for nbr in neighbours[node]:
            key = (min(node, nbr), max(node, nbr))
            node_nbrs[node].append((nbr, edge_j[key]))
    return node_nbrs


def _compute_h_for_all_zeros(
    n_points: int,
    node_nbrs: list[list[tuple[int, float]]],
    field_margin: float,
) -> list[float]:
    """Compute h_i such that all-zros (all spins -1) is a local minimum.

    Energy change flipping spin i from -1 to +1:
        delta_E = 2 * (sum_j J_ij - h_i)

    For a local minimum we need delta_E > 0 for all i, i.e.:
        h_i < sum_j J_ij

    We set h_i = sum_j J_ij - margin  (margin > 0 ensures strict inequality).
    Thisworks for any J (positive or negtive).
    """
    h = []
    for i in range(n_points):
        s = sum(j_val for _nbr, j_val in node_nbrs[i])  # sum_j J_ij
        h.append(s - field_margin)
    return h


def _gauge_transform(
    edge_j: dict[tuple[int, int], float],
    h_values: list[float],
    target: list[bool],
) -> tuple[dict[tuple[int, int], float], list[float]]:
    """Apply gauge transormation.

    J_ij -> J_ij * sigma*_i * sigma*_j,   h_i -> h_i * sigma*_i

    This maps the all-zeros (-1,-1,...) configuration of the original
    system to the configuration -sigma* in the new system, preserving
    the energy landscape (and therefore the number of local minima).
    """
    sigma = [_spin_to_int(s) for s in target]

    new_edge_j = {}
    for (a, b), j_val in edge_j.items():
        new_edge_j[(a, b)] = j_val * sigma[a] * sigma[b]

    new_h = [h_values[i] * sigma[i] for i in range(len(h_values))]

    return new_edge_j, new_h


# ---------------------------------------------------------------------------
# Enumeration (fast path for local minima counting)
# ---------------------------------------------------------------------------


def _flatten_neighbours(
    node_nbrs: list[list[tuple[int, float]]],
) -> tuple[list[int], list[float], list[int]]:
    """Flatten per-node neighbour lists into dense arrays for fast enumeration."""
    n = len(node_nbrs)
    offsets = [0] * (n + 1)
    nbr_flat = []
    j_flat = []
    for i in range(n):
        offsets[i] = len(nbr_flat)
        for nbr, j_val in node_nbrs[i]:
            nbr_flat.append(nbr)
            j_flat.append(j_val)
    offsets[n] = len(nbr_flat)
    return nbr_flat, j_flat, offsets


def _is_local_minimum(config, offsets, nbr_flat, j_flat, h_values) -> bool:
    """Check if config is a local minimum using flattened neighbour arrays."""
    for i, si in enumerate(config):
        local = 0.0
        for idx in range(offsets[i], offsets[i + 1]):
            local += j_flat[idx] * config[nbr_flat[idx]]
        if si * (local + h_values[i]) <= 0:
            return False
    return True


def _count_local_minima(
    n_points: int, node_nbrs, h_values, time_limit_s: float = 30.0
) -> int:
    """Enumerate all 2^n_points configurations; count local minima.

    Returns -1 if enumeration is aborted.
    """
    if n_points > MAX_ENUMERATION_SPINS:
        return -1

    nbr_flat, j_flat, offsets = _flatten_neighbours(node_nbrs)

    t0 = time.monotonic()
    count = 0
    total = 1 << n_points
    step = max(1, total // 100)

    config = [-1] * n_points

    for bits in range(total):
        if bits > 0:
            changed = (bits ^ (bits - 1)).bit_length() - 1
            config[changed] = -config[changed]

        # Inlined local-minimum check
        is_min = True
        for i in range(n_points):
            si = config[i]
            local = 0.0
            start = offsets[i]
            end = offsets[i + 1]
            for idx in range(start, end):
                local += j_flat[idx] * config[nbr_flat[idx]]
            if si * (local + h_values[i]) <= 0:
                is_min = False
                break
        if is_min:
            count += 1

        if bits % step == 0:
            if time.monotonic() - t0 > time_limit_s:
                return -1

    return count


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def generate_ising_file(
    dim: int,
    sizes: list[int],
    temp: float,
    mu: float,
    output_path: Path,
    seed: int | None,
    *,
    model: str = "ferro",
    j_dist: str = "normal",
    j_mean: float = 0.0,
    j_std: float = 1.0,
    j_min: float = 0.5,
    j_max: float = 2.0,
    ground_state_raw: str | None = None,
    min_local_minima: int = 0,
    max_attempts: int = 200,
    field_margin: float = 0.1,
    manual_j: float | None = None,
    manual_h: float | None = None,
    no_ground_state: bool = False,
) -> None:
    """Generate an Ising model input file.

    1. Build couplings so that all-zeros is a local minimum.
    2. If min_local_minima > 0, try random seeds until the system has at
       least that many local minima.
    3. If a ground state is reqested, gauge-transform to map all-zeros
       to the desired state.
    """
    rng = random.Random(seed)

    n_points, neighbours, edges = _build_topology(dim, sizes)

    # Manual J override
    if manual_j is not None:
        j_min = j_max = manual_j

    # Read target ground state (applied via gauge at the end)
    target: list[bool] | None = None
    if ground_state_raw is not None:
        target = _read_ground_state(ground_state_raw, n_points)

    # Too large for enumration — generate single shot imediately
    base_seed = seed if seed is not None else 0

    if n_points > MAX_ENUMERATION_SPINS:
        if min_local_minima > 0:
            print(
                f"WARNING: {n_points} spins -> 2^{n_points} configurations. "
                f"Enumeration infeasible (>2^{MAX_ENUMERATION_SPINS}). "
                f"Ignoring --min-local-minima; generating single instance."
            )
        rng.seed(base_seed)
        edge_j = _generate_edge_couplings(
            edges, model, j_dist, j_mean, j_std, j_min, j_max, rng
        )
        node_nbrs = _compute_node_neighbour_j(n_points, neighbours, edge_j)
        if manual_h is not None:
            h_values = [manual_h] * n_points
        elif no_ground_state:
            h_values = [rng.uniform(-2.0, -0.1) for _ in range(n_points)] if model == "ferro" else [0.0] * n_points
        else:
            h_values = _compute_h_for_all_zeros(n_points, node_nbrs, field_margin)
        actual_minima = -1
    else:
        # Multi-atempt: find J/h with at least min_local_minima local minima
        best_count = -1
        best_edge_j = None
        best_h = None

        for attempt in range(max_attempts if min_local_minima > 0 else 1):
            trial_seed = base_seed + attempt
            rng.seed(trial_seed)

            # Generate J couplings
            trial_j = _generate_edge_couplings(
                edges, model, j_dist, j_mean, j_std, j_min, j_max, rng
            )
            trial_nbrs = _compute_node_neighbour_j(n_points, neighbours, trial_j)

            # Compute h to make all-zeros a local minimum
            if manual_h is not None:
                trial_h = [manual_h] * n_points
            elif no_ground_state:
                trial_h = [rng.uniform(-2.0, -0.1) for _ in range(n_points)] if model == "ferro" else [0.0] * n_points
            else:
                trial_h = _compute_h_for_all_zeros(n_points, trial_nbrs, field_margin)

            # Count local minima
            count = _count_local_minima(n_points, trial_nbrs, trial_h)
            if count < 0:  # timeout
                print(f"  Enumeration timed out after {attempt + 1} attempts.")
                break

            # Track best (highest count)
            if count > best_count:
                best_count = count
                best_edge_j = dict(trial_j)
                best_h = list(trial_h)

            if min_local_minima > 0 and count >= min_local_minima:
                print(
                    f"  Found {count} local minima >= {min_local_minima} "
                    f"on attempt {attempt + 1} (seed {trial_seed})."
                )
                break
        else:
            if min_local_minima > 0 and best_count >= 0:
                verdict = "FOUND!" if best_count >= min_local_minima else "NOT met"
                print(
                    f"  Best: {best_count} local minima (need >= {min_local_minima}) "
                    f"after {max_attempts} attempts ({verdict})."
                )

        edge_j = best_edge_j
        h_values = best_h
        actual_minima = best_count

        # If all attempts timed out, generate a single shot
        if edge_j is None:
            rng.seed(base_seed)
            edge_j = _generate_edge_couplings(
                edges, model, j_dist, j_mean, j_std, j_min, j_max, rng
            )
            node_nbrs = _compute_node_neighbour_j(n_points, neighbours, edge_j)
            if manual_h is not None:
                h_values = [manual_h] * n_points
            else:
                h_values = _compute_h_for_all_zeros(n_points, node_nbrs, field_margin)
            actual_minima = -1

    # Build node_nbrs for output / reporting
    node_nbrs = _compute_node_neighbour_j(n_points, neighbours, edge_j)

    # Gauge transform if user requested a non-all-zeros ground state
    if target is not None:
        # Standard gauge J'=J*s*, h'=h*s* maps all-zeros -> -s* (complement).
        # To map all-zeros -> s* instead, apply gauge to the complement.
        target_complement = [not s for s in target]
        edge_j, h_values = _gauge_transform(edge_j, h_values, target_complement)
        node_nbrs = _compute_node_neighbour_j(n_points, neighbours, edge_j)
        # Verify the planted ground state (target) is a local minimum
        nbr_flat, j_flat, offsets = _flatten_neighbours(node_nbrs)
        cfg = [_spin_to_int(s) for s in target]
        if not _is_local_minimum(cfg, offsets, nbr_flat, j_flat, h_values):
            print("  WARNING: gauge-transormed ground state is NOT a local minimum! (bug)")

    # Build weight entry list (in Rust-reader order)
    weight_entries = []
    for a, b in edges:
        weight_entries.append(edge_j[(a, b)])

    # J stats
    j_vals = list(edge_j.values())
    j_min_actual = min(j_vals)
    j_max_actual = max(j_vals)
    n_ferro = sum(1 for v in j_vals if v > 0)
    n_anti = sum(1 for v in j_vals if v < 0)

    # Write file
    with open(output_path, "w") as f:
        f.write("$temp\n")
        f.write(f"{temp}\n\n")

        f.write("$dim_sizes\n")
        f.write(f"{dim}")
        for s in sizes:
            f.write(f" {s}")
        f.write("\n\n")

        f.write("$edge_weights_start\n")
        for w in weight_entries:
            f.write(f"{w}\n")
        f.write("\n")

        f.write("$mu\n")
        f.write(f"{mu}\n\n")

        f.write("$external_magnetic_field\n")
        for hv in h_values:
            f.write(f"{hv}\n")
        f.write("\n")

    # Report
    print(f"Generated Ising model file: {output_path}")
    print(f"  Model:         {model}" + (f" ({j_dist})" if model == "edwards-anderson" else ""))
    print(f"  Dimensions:    {dim}D, sizes: {sizes} ({n_points} total spins)")
    print(f"  Edges:         {len(edges)} undirected")
    print(f"  Couplings J:   {j_min_actual:.4f} to {j_max_actual:.4f}")
    print(f"    ferromagnetic edes:  {n_ferro}")
    print(f"    antiferromagnetic:    {n_anti}")
    if model == "edwars-anderson":
        j_mean_actual = sum(j_vals) / len(j_vals)
        j_var_actual = sum((v - j_mean_actual) ** 2 for v in j_vals) / len(j_vals)
        print(f"    actual mean:  {j_mean_actual:.4f}  (requested: {j_mean})")
        print(f"    actual std:   {math.sqrt(j_var_actual):.4f}  (requested: {j_std})")
    print(f"  Field h:       {min(h_values):.4f} to {max(h_values):.4f}")
    if target is not None:
        gs_str = "".join("1" if s else "0" for s in target)
        if len(gs_str) <= 64:
            print(f"  Ground state:  {gs_str}  (planted via gauge transform)")
        else:
            print(f"  Ground state:  [{len(gs_str)} spins, planted via gauge transform]")
    else:
        print(f"  Ground state:  all-zeros (default)")
    print(f"  Temperature:   {temp}")
    if actual_minima >= 0:
        print(f"  Local minima:  {actual_minima}")
    else:
        print(f"  Local minma:  not enumerated (system too large)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an Ising model input file."
    )

    # Grid
    parser.add_argument("-d", "--dim", type=int, default=2,
                        help="Number of dimensions (default: 2)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 4],
                        help="Grid sizes per dimnsion (default: 4 4)")

    # Model type
    parser.add_argument("--model", choices=["ferro", "edwards-anderson"],
                        default="ferro",
                        help="Coupling model: ferro (all J>0) or edwards-anderson (random J) (default: ferro)")

    # Distribution for Edwards-Anderson
    parser.add_argument("--j-dist", choices=["normal", "uniform"],
                        default="normal",
                        help="Distribution for Edwards-Anderson J (default: normal)")
    parser.add_argument("--j-mean", type=float, default=0.0,
                        help="Mean for normal J distribution (default: 0.0)")
    parser.add_argument("--j-std", type=float, default=1.0,
                        help="Std dev for normal J distribution (default: 1.0)")
    parser.add_argument("--j-min", type=float, default=None,
                        help="Min for uniform J, or min for ferro random range (default: -1.0 EA, 0.5 ferro)")
    parser.add_argument("--j-max", type=float, default=None,
                        help="Max for uniform J, or max for ferro random range (default: 1.0 EA, 2.0 ferro)")

    # Legacy flags
    parser.add_argument("-J", "--coupling", type=float, default=None,
                        help="Unifom J for all edges (overrides distribution)")
    parser.add_argument("-H", "--field", type=float, default=None,
                        help="Uniform external field h for all sites (overrides ground-state computation)")

    # Ground state & local minima
    parser.add_argument("--ground-state", type=str, default=None,
                        metavar="STR",
                        help="Target ground state as bitstring (e.g. 01001100) or path to file")
    parser.add_argument("--min-local-minima", type=int, default=0,
                        metavar="N",
                        help="Require at least N local minima (tries random seeds; bails if > 2^24)")
    parser.add_argument("--max-attempts", type=int, default=200,
                        help="Max randomseeds to try (default: 200)")
    parser.add_argument("--field-margin", type=float, default=0.1,
                        help="Margin for ground-state h computation (default: 0.1)")

    # Other
    parser.add_argument("-t", "--temp", type=float, default=1.0,
                        help="Temprature (default: 1.0)")
    parser.add_argument("-m", "--mu", type=float, default=1.0,
                        help="Magnetic moment scaling factor (default: 1.0)")
    parser.add_argument("-o", "--output", type=Path, default=Path("ising_input.txt"),
                        help="Output file path (default: ising_input.txt)")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Random seed for reproducible generation")

    args = parser.parse_args()

    # Validate
    if args.dim < 1:
        parser.error("--dim must be >= 1")
    if len(args.sizes) != args.dim:
        parser.error(
            f"--sizes needs exactly {args.dim} values (got {len(args.sizes)}: {args.sizes})"
        )
    if any(s < 1 for s in args.sizes):
        parser.error("All sizes must be >= 1")

    # Resolve j_min/j_max defaults based on model
    if args.j_min is None:
        args.j_min = 0.5 if args.model == "ferro" else -1.0
    if args.j_max is None:
        args.j_max = 2.0 if args.model == "ferro" else 1.0

    generate_ising_file(
        dim=args.dim,
        sizes=args.sizes,
        temp=args.temp,
        mu=args.mu,
        output_path=args.output,
        seed=args.seed,
        model=args.model,
        j_dist=args.j_dist,
        j_mean=args.j_mean,
        j_std=args.j_std,
        j_min=args.j_min,
        j_max=args.j_max,
        ground_state_raw=args.ground_state,
        min_local_minima=args.min_local_minima,
        max_attempts=args.max_attempts,
        field_margin=args.field_margin,
        manual_j=args.coupling,
        manual_h=args.field,
    )


if __name__ == "__main__":
    cli()