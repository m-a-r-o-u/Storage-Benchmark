"""PyTorch-powered storage I/O benchmark.

This script creates synthetic tensor datasets on a given set of storage
locations and measures sequential write/read throughput using PyTorch I/O.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class BenchmarkResult:
    target: Path
    run_dir: Path
    num_files: int
    total_bytes: int
    write_samples: List[float] = field(default_factory=list)
    read_samples: List[float] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return len(self.write_samples)

    @property
    def write_seconds(self) -> float:
        if not self.write_samples:
            raise RuntimeError("No write samples recorded")
        return statistics.median(self.write_samples)

    @property
    def read_seconds(self) -> float:
        if not self.read_samples:
            raise RuntimeError("No read samples recorded")
        return statistics.median(self.read_samples)

    @property
    def write_throughput_gbps(self) -> float:
        return (self.total_bytes * 8) / (self.write_seconds * 1e9)

    @property
    def read_throughput_gbps(self) -> float:
        return (self.total_bytes * 8) / (self.read_seconds * 1e9)


class TensorFileDataset(Dataset):
    """Dataset that lazily loads tensors saved as individual files."""

    def __init__(self, files: Sequence[Path], map_location: torch.device | str):
        self._files: List[Path] = list(files)
        self._sizes: List[int] = [f.stat().st_size for f in self._files]
        self._map_location = map_location

    def __len__(self) -> int:  # noqa: D401 - simple delegation
        return len(self._files)

    @property
    def total_bytes(self) -> int:
        return sum(self._sizes)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_path = self._files[index]
        return torch.load(file_path, map_location=self._map_location)


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(f"Unknown dtype: {name}") from exc
    if not isinstance(dtype, torch.dtype):  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(f"Attribute torch.{name} is not a dtype")
    return dtype


def generate_random_tensor(num_elements: int, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(num_elements, dtype=dtype, generator=generator)


def aligned_total_bytes(total_bytes: int, element_size: int) -> int:
    return math.ceil(total_bytes / element_size) * element_size


def ensure_unique_directory(base_dir: Path, prefix: str) -> Path:
    for attempt in range(100):
        timestamped = f"{prefix}_{int(time.time())}_{attempt}"
        candidate = base_dir / timestamped
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("Unable to create a unique benchmark directory")


def flush_os_buffers() -> None:
    """Ensure pending writes reach the storage device."""

    sync = getattr(os, "sync", None)
    if callable(sync):
        sync()


def drop_linux_page_cache() -> bool:
    """Attempt to evict the Linux page cache to avoid warm reads."""

    drop_path = Path("/proc/sys/vm/drop_caches")
    if not drop_path.exists():
        return False

    flush_os_buffers()
    try:
        drop_path.write_text("3\n")
    except PermissionError:
        print(
            "Warning: unable to drop caches (permission denied). "
            "Run with elevated privileges to enable --drop-caches.",
            file=sys.stderr,
        )
        return False
    except OSError:
        return False
    return True


def write_dataset(
    target: Path,
    total_size_gb: float,
    chunk_size_mb: int,
    dtype: torch.dtype,
    seed: int,
    fsync: bool,
) -> tuple[List[Path], int, float]:
    generator = torch.Generator().manual_seed(seed)
    element_size = torch.tensor([], dtype=dtype).element_size()
    requested_bytes = int(total_size_gb * (1024**3))
    aligned_bytes = aligned_total_bytes(requested_bytes, element_size)

    raw_chunk_bytes = int(chunk_size_mb * 1024 * 1024)
    elements_per_chunk = max(raw_chunk_bytes // element_size, 1)
    chunk_bytes = elements_per_chunk * element_size
    num_chunks = math.ceil(aligned_bytes / chunk_bytes)

    files: List[Path] = []
    bytes_written = 0

    start = time.perf_counter()
    for chunk_index in range(num_chunks):
        remaining_bytes = aligned_bytes - bytes_written
        current_chunk_bytes = min(chunk_bytes, remaining_bytes)
        elements_in_chunk = max(current_chunk_bytes // element_size, 1)
        tensor = generate_random_tensor(elements_in_chunk, dtype=dtype, generator=generator)

        file_path = target / f"tensor_{chunk_index:05d}.pt"
        with open(file_path, "wb") as handle:
            torch.save(tensor, handle)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        files.append(file_path)
        bytes_written += file_path.stat().st_size
    elapsed = time.perf_counter() - start
    return files, bytes_written, elapsed


def read_dataset(
    files: Sequence[Path],
    num_workers: int,
    map_location: torch.device | str,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    transfer_to_device: bool,
    device: torch.device,
) -> tuple[int, float]:
    dataset = TensorFileDataset(files, map_location=map_location)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    loader = DataLoader(**loader_kwargs)

    total_samples = len(dataset)
    total_bytes = dataset.total_bytes

    start = time.perf_counter()
    for tensor in loader:
        if transfer_to_device:
            tensor = tensor.to(device, non_blocking=pin_memory)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        del tensor
    elapsed = time.perf_counter() - start

    if total_samples == 0:
        raise RuntimeError("No tensor files were generated for reading")
    return total_bytes, elapsed


def benchmark_target(
    target: Path,
    args: argparse.Namespace,
) -> BenchmarkResult:
    target = target.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target}")

    run_dir = ensure_unique_directory(target, prefix="storage_benchmark")
    write_samples: List[float] = []
    read_samples: List[float] = []
    num_files = 0
    total_bytes = 0
    drop_warning_printed = False

    try:
        for sample_index in range(args.samples):
            sample_dir = run_dir / f"sample_{sample_index:02d}"
            sample_dir.mkdir(parents=True, exist_ok=False)

            files, bytes_written, write_seconds = write_dataset(
                sample_dir,
                total_size_gb=args.total_size_gb,
                chunk_size_mb=args.chunk_size_mb,
                dtype=args.dtype,
                seed=args.seed + sample_index,
                fsync=args.fsync,
            )

            flush_os_buffers()

            drop_success = False
            if args.drop_caches:
                drop_success = drop_linux_page_cache()
                if not drop_success and not drop_warning_printed:
                    print(
                        "Warning: could not drop caches; read benchmark may still "
                        "hit the page cache.",
                        file=sys.stderr,
                    )
                    drop_warning_printed = True

            total_bytes, read_seconds = read_dataset(
                files,
                num_workers=args.num_workers,
                map_location=torch.device("cpu"),
                pin_memory=args.pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                transfer_to_device=args.transfer_to_device,
                device=args.device,
            )

            write_samples.append(write_seconds)
            read_samples.append(read_seconds)
            num_files = len(files)

            if bytes_written != total_bytes:
                raise RuntimeError(
                    "Mismatch between written and read bytes: "
                    f"{bytes_written} vs {total_bytes}"
                )

            write_gbps = (bytes_written * 8) / (write_seconds * 1e9)
            read_gbps = (total_bytes * 8) / (read_seconds * 1e9)
            print(
                f"  Sample {sample_index + 1}/{args.samples}: "
                f"write {write_seconds:.2f}s ({write_gbps:.2f} Gb/s), "
                f"read {read_seconds:.2f}s ({read_gbps:.2f} Gb/s)"
            )

            if not args.keep_data:
                shutil.rmtree(sample_dir, ignore_errors=True)
    finally:
        if not args.keep_data:
            shutil.rmtree(run_dir, ignore_errors=True)

    return BenchmarkResult(
        target=target,
        run_dir=run_dir,
        num_files=num_files,
        total_bytes=total_bytes,
        write_samples=write_samples,
        read_samples=read_samples,
    )


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = units[0]
    for next_unit in units[1:]:
        if value < 1024.0:
            break
        value /= 1024.0
        unit = next_unit
    return f"{value:.2f} {unit}"


def summarize_results(results: Iterable[BenchmarkResult]) -> None:
    results = list(results)

    print("\nBenchmark summary:\n")

    if not results:
        print("No benchmark results to display.\n")
        return

    print("Targets:")
    for index, result in enumerate(results, start=1):
        print(f"  {index}: {result.target}")
    print()

    rows = []
    for index, result in enumerate(results, start=1):
        rows.append(
            {
                "target": str(index),
                "data_size": format_bytes(result.total_bytes),
                "files": str(result.num_files),
                "samples": str(result.num_samples),
                "write_seconds": f"{result.write_seconds:.2f}",
                "write_gbps": f"{result.write_throughput_gbps:.2f}",
                "read_seconds": f"{result.read_seconds:.2f}",
                "read_gbps": f"{result.read_throughput_gbps:.2f}",
            }
        )

    columns = [
        ("Target #", "target", ">"),
        ("Data Size", "data_size", ">"),
        ("Files", "files", ">"),
        ("Samples", "samples", ">"),
        ("Write (s)", "write_seconds", ">"),
        ("Write (Gb/s)", "write_gbps", ">"),
        ("Read (s)", "read_seconds", ">"),
        ("Read (Gb/s)", "read_gbps", ">"),
    ]

    widths = {
        key: max(len(header), *(len(row[key]) for row in rows))
        for header, key, _ in columns
    }

    def format_line(values: dict[str, str]) -> str:
        return "  ".join(
            f"{values[key]:{align}{widths[key]}}" for _, key, align in columns
        )

    header_line = "  ".join(
        f"{header:{align}{widths[key]}}" for header, key, align in columns
    )
    separator = "-" * len(header_line)

    print(header_line)
    print(separator)
    for row in rows:
        print(format_line(row))
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch storage I/O benchmark")
    parser.add_argument(
        "targets",
        nargs="+",
        help="List of directories to benchmark (e.g. /mnt/nvme /mnt/lustre)",
    )
    parser.add_argument(
        "--total-size-gb",
        type=float,
        default=2.0,
        help="Total amount of tensor data to write/read per target (in GiB).",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=256,
        help="Size of each tensor file (in MiB).",
    )
    parser.add_argument(
        "--dtype",
        type=parse_dtype,
        default=torch.float32,
        help="PyTorch dtype for generated tensors (e.g. float32, float16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible tensor generation.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of write/read cycles to run per target (median is reported).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for the read benchmark.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Prefetch factor for DataLoader workers (ignored when num_workers=0).",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs for read benchmark.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pinned memory in the read DataLoader.",
    )
    parser.add_argument(
        "--transfer-to-device",
        action="store_true",
        help="After loading a tensor, transfer it to the target device to include DMA in measurements.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use when --transfer-to-device is set (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--fsync",
        action="store_true",
        help="Call fsync after every write to reduce cache effects.",
    )
    parser.add_argument(
        "--drop-caches",
        action="store_true",
        help=(
            "Attempt to drop the OS page cache before each read pass (Linux only, "
            "requires elevated privileges)."
        ),
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Do not delete generated data after benchmarking.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.total_size_gb <= 0:
        raise SystemExit("--total-size-gb must be positive")
    if args.chunk_size_mb <= 0:
        raise SystemExit("--chunk-size-mb must be positive")
    if args.num_workers < 0:
        raise SystemExit("--num-workers cannot be negative")
    if args.prefetch_factor is not None and args.prefetch_factor <= 0:
        raise SystemExit("--prefetch-factor must be positive")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but CUDA is not available")
    args.device = device


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    targets = [Path(t) for t in args.targets]

    results: List[BenchmarkResult] = []
    for target in targets:
        print(f"\nBenchmarking target: {target}")
        try:
            result = benchmark_target(target, args)
        except Exception as exc:  # pragma: no cover - CLI user feedback
            print(f"Failed to benchmark {target}: {exc}", file=sys.stderr)
            return 1
        results.append(result)

    summarize_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
