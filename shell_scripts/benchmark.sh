usage() {
  echo "Usage:$0 <ising_file_name> [-r rust_ising_binary] [-T temperature] [-n steps] [-N runs] [-R repeats]"
  exit 1
}

[[ $# -ge 1 ]] || usage
ising_file_name="$1"
shift

[[ -f "$ising_file_name" ]] || {
  echo "$ising_file_name does not exist" >&2; exit 1
}

rust_ising_binary_dir="$HOME/rust_ising_and_python_tools/rust_ising"
rust_ising_binary="$rust_ising_binary_dir/target/release/rust_ising"
flags=()
steps="1000"
R="500"
while getopts ":r:T:n:N:R:" opt; do
  case "$opt" in 
    r) rust_ising_binary="$OPTARG" ;;
    T) flags+=(-T "$OPTARG") ; T="$OPTARG" ;;
    n) flags+=(-n "$OPTARG");;
    N) flags+=(-r "$OPTARG")  ;;
    R) R="$OPTARG"  ;;
    :) echo "option -$OPTARG requires an argument" >&2; usage;;
    \?) echo "unknown option: -$OPTARG >&2"; usage;;
  esac
done

tmp=~/tmp/benchmark_temp_"$T"

hyperfine --conclude "rm -f stationary_hits_finite.txt" \
  --warmup 10 \
  --runs "$R" \
  --export-json "$tmp" \
  --style none \
  -N \
  "$rust_ising_binary $ising_file_name finite-t ${flags[*]}"
jq '.results[0].summary.time_wall_clock.mean' "$tmp"
rm $tmp
