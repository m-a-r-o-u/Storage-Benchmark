# Storage-Benchmark

This repository provides a PyTorch-based workload that stresses file system
throughput by generating synthetic tensor datasets, writing them to storage,
and streaming them back using the same data loading primitives found in
machine learning training loops. The goal is to help compare the read and
write performance of different storage backends that may be used in ML/AI HPC
clusters (e.g., NVMe, Lustre, object storage gateways).

## Requirements

- Python 3.9 or newer
- [PyTorch](https://pytorch.org/) with the desired CPU/GPU support installed

Install PyTorch with the configuration that matches your hardware. For example:

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
```

## Usage

Run the benchmark by pointing it at one or more directories that reside on the
storage targets you wish to measure:

```bash
python storage_benchmark.py /mnt/nvme0 /mnt/lustre/share
```

Key options:

- `--total-size-gb`: Total volume of tensor data (in GiB) to write and read per
  target. Defaults to `2.0` GiB.
- `--chunk-size-mb`: Size (in MiB) of each tensor shard saved to disk. Larger
  chunks reduce metadata pressure while smaller chunks highlight small-file
  performance.
- `--seed`: Random seed used when synthesizing the tensor data.
- `--samples`: Run multiple write/read cycles per target and report the median
  throughput. Increases runtime but reduces noise from transient effects.
- `--num-workers`: Number of PyTorch `DataLoader` workers used during the read
  phase. Increase this to simulate high parallelism.
- `--pin-memory`: Enable pinned host memory for the read loader. Combine with
  `--transfer-to-device` to include host→device DMA time in the measurement.
- `--transfer-to-device` and `--device`: After loading each tensor, move it to
  the specified device (e.g., `cuda:0`) to capture end-to-end throughput.
- `--fsync`: Issue an `fsync` call after writing each shard to minimize the
  impact of page cache effects.
- `--drop-caches`: Attempt to flush and drop the OS page cache before each read
  phase (Linux only, requires elevated privileges). This makes the read
  benchmark much more indicative of the underlying storage hardware instead of
  serving data from RAM.
- `--keep-data`: Preserve the generated dataset so that you can run additional
  experiments (e.g., repeat the read benchmark separately).

The script prints a concise summary with per-target write/read durations and
Gb/s throughput. You can run the benchmark multiple times—using different
chunk sizes, worker counts, or storage mount points—to build a comparison table
for your infrastructure.

### Run the recommended scenarios automatically

A convenience wrapper, [`run_benchmarks.sh`](./run_benchmarks.sh), executes the
five recommended exercises sequentially. The script prints a short description
of each scenario, the exact `python storage_benchmark.py` command that is run,
and the benchmark results. Output is streamed through `tee`, so you see the
progress live while it is also appended to a log file for later inspection.

```bash
./run_benchmarks.sh --total-size-gb 25 ./ai ./lc ./home
```

Key behavior:

- `--total-size-gb` is optional; when omitted the script discovers the default
  value from `storage_benchmark.py` (currently `2.0`).
- Use `--log-file results/benchmark.log` to place the combined output in a
  specific location. Otherwise a timestamped file such as
  `benchmark_runs_20240101_120000.log` is created in the current directory.
- Provide one or more target directories as positional arguments. The same set
  of targets is used for every scenario.
- The final "Cold-cache verification" scenario passes `--drop-caches`, which
  requires elevated privileges on Linux. Run the wrapper with the necessary
  permissions (e.g., `sudo ./run_benchmarks.sh ...`) to exercise that test fully.

The wrapper runs the following scenarios with your chosen targets:

1. Baseline throughput comparison across targets.
2. Metadata-pressure sensitivity using 8 MiB chunks.
3. Parallel read stress with 16 loader workers and three samples.
4. End-to-end GPU ingest measurement with pinned memory and `cuda:0`.
5. Cold-cache verification with `--fsync` and `--drop-caches`.

## Notes

- The script automatically synchronizes pending writes before the read pass so
  that throughput reflects the storage device instead of buffered writes.
- Use `--drop-caches` (or manually clear the OS caches) between passes to avoid
  measuring RAM-backed reads. Without this, SSD and HDD results may look
  artificially similar because the reads are served from the page cache.
- When benchmarking remote or burst-buffer filesystems, consider increasing the
  total data volume (`--total-size-gb`) to amortize initial warm-up costs.
- If GPU DMA is included, ensure that other workloads are not contending for the
  same device to avoid skewing the results.
