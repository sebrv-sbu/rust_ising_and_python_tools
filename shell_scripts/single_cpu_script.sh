set -eou pipefail

usage() {
  echo "Usage: $0 <ising_file_name> [-r rust_ising_and_python_tools_dir] [-T temperature] [-n steps] [-N runs] [-t tmp_dir] [-E err_and_warning_dir] [-M mergefile]" >&2
  echo "Warning: this file is designed only to be invoked by srun during a slurm run. Are you sure you should be running this?" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage
ising_file_name="$1"
shift
[[ -f "$ising_file_name" ]] || {
  echo "$ising_file_name does not exist" >&2; exit 1
}


flags=()
while getopts ":r:T:N:n:t:E:M:" opt; do 
  case "$opt" in 
    r) CODEDIR="$OPTARG"                      ;;
    T) flags+=(-T "$OPTARG"); T="$OPTARG"     ;;
    N) flags+=(-r "$OPTARG")                  ;;
    n) flags+=(-n "$OPTARG"); steps="$OPTARG" ;;
    t) ISINGTMPDIR="$OPTARG"                  ;;
    E) ERR_AND_WARNING_DIR="$OPTARG"          ;;
    M) MERGEFILE="$OPTARG"                    ;;
    :) echo "option -$OPTARG requires an argument" >&2; usage ;;
    \?) echo "unknown option: -$OPTARG" >&2; usage ;;
  esac
done

#Do not modify unless the rust ising directory is renamed for some odd reason
RUSTDIR=$CODEDIR/rust_ising/
BINARYDIR=$RUSTDIR/target/release
BINARY=$BINARYDIR/rust_ising

#Do not modify unless the python directory is renamed for some odd reason
SCRIPTDIR=$CODEDIR/shell_scripts
RANDOMISESCRIPT=$SCRIPTDIR/randomise.sh
LOCKSCRIPT=$SCRIPTDIR/append_locked.sh

mkdir -p "$ERR_AND_WARNING_DIR"

"$RANDOMISESCRIPT" "$ising_file_name" "$ising_file_name"_${SLURM_PROCID}_${steps}_${T}
ISINGFILE="$ising_file_name"_${SLURM_PROCID}_${steps}_${T}


"$BINARY" "$ISINGFILE" finite-t \
  -s "$ISINGTMPDIR"/tmp_run_${SLURM_PROCID} ${flags[@]} \
  2>>"$ERR_AND_WARNING_DIR/"err_output_${SLURM_PROCID} 

"$LOCKSCRIPT" \
  "$ISINGTMPDIR"/append.lock_${steps}_${T} \
  "$ISINGTMPDIR"/tmp_run_${SLURM_PROCID} \
  "${MERGEFILE}_${steps}_${T}" 

rm "$ISINGTMPDIR"/tmp_run_${SLURM_PROCID}
rm $ISINGTMPDIR/append.lock_${steps}_${T}
