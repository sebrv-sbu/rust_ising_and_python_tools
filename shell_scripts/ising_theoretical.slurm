#!/bin/bash
#SBATCH --job-name="sc4_di3"
#SBATCH --partition="long-96core"
#SBATCH --nodes=6
#SBATCH --ntasks=576
#SBATCH --time=48:00:00

#Modify to be the code directory you will use here:
CODEDIR=~/rust_ising_and_python_tools3

#Do not modify unless the rust ising directory is renamed for some odd reason
RUSTDIR=$CODEDIR/rust_ising

module load rust
module load python

cd $RUSTDIR

DIM=2
cargo clean
ISING_DIM=$DIM RUSTFLAGS="-C target-cpu=native" cargo build --release

#Do not modify these unless you are running debug mode, then change
#release to debug in BINARYDIR.
BINARYDIR=$RUSTDIR/target/release
BINARY=$BINARYDIR/rust_ising

#Do not modify unless the python directory is renamed for some odd reason
PYTHONDIR=$CODEDIR/python_tools
SCRIPTDIR=$CODEDIR/shell_scripts
chmod +x $SCRIPTDIR/*sh
RANDOMISESCRIPT=$SCRIPTDIR/randomise.sh
SINGLECPU=$SCRIPTDIR/single_cpu_script.sh
BENCHMARK=$SCRIPTDIR/benchmark.sh

#Modify accordingly
ISINGDIR="/gpfs/scratch/sebramirezvi/ising/theoretical/6x6_low_temp"
ERR_AND_WARNING=$ISINGDIR/err_output
ISINGTMPDIR=$ISINGDIR/tmp
ERR_AND_WARNING_DIR=$ISINGDIR/proc_errs
mkdir -p $ERR_AND_WARNING_DIR
mkdir -p $ISINGTMPDIR

cd $ISINGDIR
#Modify to your specifications
ISINGFILE=$ISINGDIR/inst6x6

TEMPS=(3e5 1e6 3e6 1e7 3e-5 1e-5)


for T in "${TEMPS[@]}"; do
  below() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }
  time_per_100000=$("$BENCHMARK" "$ISINGFILE" -T "$T" -R 1000 -n 100000 -N 1)
  if below "$T" 0.5 ; then
    mode=low-conv
  else
    mode=high-conv
  fi

  ten_percent_steps=$("$BINARY" "$ISINGFILE" "$mode" -T "$T" -p 0.95 -t \
   | tail -n 1 \
   | awk -F '[ ,]+' '{ print $1 }')

  if below $ten_percent_steps 1000000; then
    ten_percent_steps=1000000
  fi

  time_per_run=$(awk -v t="$time_per_100000" -v s="$ten_percent_steps" \
    'BEGIN { print t * s / 100000 }')

  echo $ten_percent_steps
  echo $time_per_run
  min_runs=12
  total_hours=48

  steps=$(awk -v t="$time_per_100000" -v s="$ten_percent_steps" \
                -v hours="$total_hours" -v runs="$min_runs" '
    BEGIN {
    if (t <= 0) {
      printf "%.0f\n", s
    } else {
      budget = hours * 3600 / runs          # seconds available per run
      max_steps = budget * 10000000 / t     # steps that fit in that budget
      printf "%.0f\n", (max_steps < s) ? max_steps : s
    }
  }')
  echo $steps
    srun --exact --ntasks=96 --cpus-per-task=1 \
      "$SINGLECPU" "$ISINGFILE" \
      -r "$CODEDIR" -T "$T" -n "$steps" -N "4294967295" -t "$ISINGTMPDIR" \
      -E "$ERR_AND_WARNING_DIR" -M "$ISINGDIR/final_output_T_${T}"&
done

wait
