#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_benchmarks.sh [OPTIONS] TARGET [TARGET ...]

Run the recommended storage benchmark scenarios sequentially and capture
output to both the terminal and a log file.

Options:
  --total-size-gb VALUE  Total dataset size (GiB) for write/read tests.
  --log-file PATH        File to append combined output (default includes timestamp).
  -h, --help             Show this help message and exit.

Example:
  ./run_benchmarks.sh --total-size-gb 25 ./ai ./lc ./home
USAGE
}

read_default_total_size() {
  python - <<'PY'
from pathlib import Path
import ast

source = Path("storage_benchmark.py").read_text()
tree = ast.parse(source, filename="storage_benchmark.py")

class DefaultFinder(ast.NodeVisitor):
    def __init__(self):
        self.value = None

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--total-size-gb"
        ):
            for keyword in node.keywords:
                if keyword.arg == "default":
                    self.value = ast.literal_eval(keyword.value)
        self.generic_visit(node)

finder = DefaultFinder()
finder.visit(tree)

if finder.value is None:
    raise SystemExit("Unable to determine --total-size-gb default from storage_benchmark.py")

print(finder.value)
PY
}

DEFAULT_TOTAL_SIZE_GB=$(read_default_total_size)
total_size_gb="$DEFAULT_TOTAL_SIZE_GB"
log_file=""

targets=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --total-size-gb)
      [[ $# -ge 2 ]] || { echo "Missing value for --total-size-gb" >&2; exit 1; }
      total_size_gb="$2"
      shift 2
      ;;
    --log-file)
      [[ $# -ge 2 ]] || { echo "Missing value for --log-file" >&2; exit 1; }
      log_file="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        targets+=("$1")
        shift
      done
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      targets+=("$1")
      shift
      ;;
  esac
done

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "At least one target directory must be provided." >&2
  usage
  exit 1
fi

if [[ -z "$log_file" ]]; then
  timestamp=$(date +%Y%m%d_%H%M%S)
  log_file="benchmark_runs_${timestamp}.log"
fi

log_dir=$(dirname "$log_file")
if [[ -n "$log_dir" && "$log_dir" != "." ]]; then
  mkdir -p "$log_dir"
fi

echo "Writing combined output to: $log_file"

exec > >(tee -a "$log_file")
exec 2>&1

run_test() {
  local description="$1"
  shift
  local cmd=("$@")

  echo
  echo "======================================================================"
  echo "### $description"
  printf '# Command:'
  printf ' %q' "${cmd[@]}"
  echo
  echo "----------------------------------------------------------------------"

  set +e
  "${cmd[@]}"
  local status=$?
  set -e

  echo "----------------------------------------------------------------------"
  if [[ $status -eq 0 ]]; then
    echo "# Result: success"
  else
    echo "# Result: failure (exit code $status)" >&2
    exit $status
  fi
}

run_test "Baseline throughput comparison across targets" \
  python storage_benchmark.py "${targets[@]}" --total-size-gb "$total_size_gb"

run_test "Metadata-pressure sensitivity via small shards" \
  python storage_benchmark.py "${targets[@]}" --total-size-gb "$total_size_gb" --chunk-size-mb 8

run_test "Parallel read stress with many loader workers" \
  python storage_benchmark.py "${targets[@]}" --total-size-gb "$total_size_gb" --samples 3 --num-workers 16

run_test "End-to-end GPU ingest measurement" \
  python storage_benchmark.py "${targets[@]}" --total-size-gb "$total_size_gb" --transfer-to-device --device cuda:0 --pin-memory

run_test "Cold-cache verification of storage hardware limits" \
  python storage_benchmark.py "${targets[@]}" --total-size-gb "$total_size_gb" --fsync --drop-caches

