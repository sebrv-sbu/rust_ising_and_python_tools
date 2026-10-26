#!/bin/bash
#SBATCH --job-name="lt1_d1"
#SBATCH --partition="long-96core"
#SBATCH --nodes=6
#SBATCH --ntasks=576
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2G

#Modify to be the code directory you will use here:
CODEDIR=~/rust_ising_and_python_tools

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

TEMPS=(3e-5 1e-5 3e-4 1e-4 3e-3 1e-3)


for T in "${TEMPS[@]}"; do
  below() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }
  above() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'; }
  if below "$T" 0.5 ; then
    mode=low-conv
  else
    mode=high-conv
  fi

  steps=$("$BINARY" "$ISINGFILE" "$mode" -T "$T" -p 0.5 -t \
   | tail -n 1 \
   | awk -F '[ ,]+' '{ print $1 }')

  if below $steps 1000000; then
    steps=1000000
  fi

  if above "$steps" 2.8e12; then
    steps=2800000000000
  fi

  echo $steps
    srun --exact --ntasks=96 --mem-per-cpu=2G \
      "$SINGLECPU" "$ISINGFILE" \
      -r "$CODEDIR" -T "$T" -n "$steps" -N "4294967295" -t "$ISINGTMPDIR" \
      -E "$ERR_AND_WARNING_DIR" -M "$ISINGDIR/final_output_T_${T}"&
done
wait
