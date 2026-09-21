#!/usr/bin/env bash
# reseed.sh: copy an input file with a fresh OS-generated $rng_seed
# usage: ./reseed.sh input.txt [output.txt]
set -euo pipefail

in="${1:?usage: $0 input [output]}"
[[ -f "$in" ]] || { echo "No such file: $in" >&2; exit 1; }
grep -q '^\$rng_seed[[:space:]]*$' "$in" \
  || { echo "No \$rng_seed section in $in" >&2; exit 1; }

# 8 random bytes from the OS -> 16 hex digits (fits in u64)
seed=$(od -An -N8 -tx8 /dev/urandom | tr -d ' \n')

if [[ $# -ge 2 ]]; then
  out="$2"
elif [[ "$in" == *.* ]]; then
  out="${in%.*}_${seed}.${in##*.}"
else
  out="${in}_${seed}"
fi

[[ -e "$out" ]] && { echo "Refusing to overwrite $out" >&2; exit 1; }

awk -v seed="$seed" '
  found && NF == 0 { print; next }            # keep blank lines before the value
  found            { print seed; found = 0; next }
  /^\$rng_seed[[:space:]]*$/ { found = 1 }
  { print }
' "$in" > "$out"

echo "seed $seed -> $out"
