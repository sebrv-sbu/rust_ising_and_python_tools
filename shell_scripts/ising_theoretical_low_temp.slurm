#!/bin/bash
#SBATCH --job-name="l1_d1_3d"
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

DIM=3
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
ISINGDIR="/gpfs/scratch/sebramirezvi/ising/theoretical/4x4x4"
ERR_AND_WARNING=$ISINGDIR/err_output
ISINGTMPDIR=$ISINGDIR/tmp
ISINGCHECKPOINTDIR=$ISINGDIR/checkpoints
ERR_AND_WARNING_DIR=$ISINGDIR/proc_errs
mkdir -p $ERR_AND_WARNING_DIR
mkdir -p $ISINGTMPDIR
mkdir -p $ISINGCHECKPOINTDIR

cd $ISINGDIR
#Modify to your specifications
ISINGFILE=$ISINGDIR/inst_4x4x4

TEMPS=(1.8e-1 4.1e0 4.6e0 5e0 6e0 6.8e0)


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

  if below $steps 200000000000; then
    steps=200000000000
  fi

  if above "$steps" 2.8e12; then
    steps=2800000000000
  fi

  echo $steps
    srun --exact --ntasks=96 --mem-per-cpu=2G \
      "$SINGLECPU" "$ISINGFILE" \
      -r "$CODEDIR" -T "$T" -n "$steps" -N "4294967295" -t "$ISINGTMPDIR" \
      -E "$ERR_AND_WARNING_DIR" -M "$ISINGDIR/final_output_T_${T}" \
      -c "$ISINGCHECKPOINTDIR"&
done
wait
