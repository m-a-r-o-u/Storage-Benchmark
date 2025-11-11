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
- `--num-workers`: Number of PyTorch `DataLoader` workers used during the read
  phase. Increase this to simulate high parallelism.
- `--pin-memory`: Enable pinned host memory for the read loader. Combine with
  `--transfer-to-device` to include host→device DMA time in the measurement.
- `--transfer-to-device` and `--device`: After loading each tensor, move it to
  the specified device (e.g., `cuda:0`) to capture end-to-end throughput.
- `--fsync`: Issue an `fsync` call after writing each shard to minimize the
  impact of page cache effects.
- `--keep-data`: Preserve the generated dataset so that you can run additional
  experiments (e.g., repeat the read benchmark separately).

The script prints a concise summary with per-target write/read durations and
Gb/s throughput. You can run the benchmark multiple times—using different
chunk sizes, worker counts, or storage mount points—to build a comparison table
for your infrastructure.

## Notes

- Clearing the operating system page cache between read passes (outside the
  script) can help isolate storage hardware performance from cache effects.
- When benchmarking remote or burst-buffer filesystems, consider increasing the
  total data volume (`--total-size-gb`) to amortize initial warm-up costs.
- If GPU DMA is included, ensure that other workloads are not contending for the
  same device to avoid skewing the results.
