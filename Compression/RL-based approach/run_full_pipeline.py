from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path
from compressor_runner import (decompress_bytes_blocks, compress_nucleotides_with_trained_rl, decompress_nucleotides_with_metadata)
from feature_cache import build_window_feature_cache


MAGIC = b"RLC"
DEFAULT_WINDOW_SIZE = 1000
DEFAULT_SEGMENT_LENGTHS = [200000, 400000, 600000, 800000]
DEFAULT_RL_MODEL = "pretrained_dqn_model_2.pt"


def _build_feature_cache_from_sequence(current_nucleotides: str, window_size: int) -> list[list[float]]:
    """
    Builds temporary feature cache directly from the sequence string.
    No temporary FASTA file is needed for feature extraction.
    """
    start = time.time()

    window_features = build_window_feature_cache(current_nucleotides, window_size=window_size)

    print(f"[features extraction time] {time.time() - start}s")
    # print(f"[features] num windows: {len(window_features)}")
    return window_features


def _resolve_existing_path(path: str | Path, script_dir: Path) -> Path:
    """
    Resolves model path.

    First tries the path as given.
    If it does not exist, tries it relative to the script directory.
    """

    path = Path(path)

    if path.exists():
        return path

    candidate = script_dir / path

    if candidate.exists():
        return candidate

    return path


def _sequence_index(path: Path) -> int:
    """
    Extracts the numeric index from filenames like:
        sequence_1
        sequence_2
        sequence_10

    This avoids wrong lexicographic order:
        sequence_1, sequence_10, sequence_2
    """

    match = re.fullmatch(r"sequence_(\d+)", path.name)

    if not match:
        raise ValueError(f"Invalid compressed sequence filename: {path.name}")

    return int(match.group(1))


def _compute_compression_metrics_vs_file(current_nucleotides: str, output_path: Path,) -> dict[str, float]:
    """
    Compares one compressed payload against the original nucleotide string.
    Here, output_path is a compressed file, not a folder.
    """

    output_path = Path(output_path)

    original_size = len(current_nucleotides.encode("ascii"))
    compressed_size = output_path.stat().st_size

    if original_size == 0:
        raise ValueError("Original sequence is empty!")

    compression_rate = compressed_size / original_size
    compression_gain = 1.0 - compression_rate
    compression_ratio = original_size / compressed_size if compressed_size else 0.0

    return {
        "original_payload_size_bytes": original_size,
        "compressed_payload_size_bytes": compressed_size,
        "compression_rate": compression_rate,
        "compression_gain": compression_gain,
        "compression_ratio": compression_ratio,
    }


def run_compression_pipeline(
        mode: str,
        current_nucleotides: str,
        output_path: str | Path,
        rl_model: str | Path = DEFAULT_RL_MODEL,
        window_size: int = DEFAULT_WINDOW_SIZE,
        segment_lengths: list[int] | None = None,
        parallel: bool = False,
        workers: int | None = None,
        device: str | None = None,
        overwrite: bool = False,
        line_width: int=60
) -> None :

    if segment_lengths is None:
        segment_lengths = DEFAULT_SEGMENT_LENGTHS

    mode = mode.lower().strip()

    if mode not in {"c", "compress", "d", "decompress"}:
        raise ValueError("mode must be one of: c, compress, d, decompress")

    output_path = Path(output_path)

    script_dir = Path(__file__).resolve().parent
    rl_model = _resolve_existing_path(rl_model, script_dir)


    #-----------------------------------------COMPRESSION--------------------------------------------

    if mode in {"c", "compress"}:

        if not rl_model.exists():
            raise FileNotFoundError(f"RL model not found: {rl_model}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            if output_path.is_dir():
                if overwrite:
                    shutil.rmtree(output_path)
                else:
                    raise IsADirectoryError(f"Output path already exists as a directory: {output_path}")
            else:
                if overwrite:
                    output_path.unlink()
                else:
                    raise FileExistsError(f"Output compressed file already exists: {output_path}")

        print("\n==============================")
        print("Running compression pipeline")
        print("==============================")

        # Build feature cache directly

        window_features = _build_feature_cache_from_sequence(current_nucleotides=current_nucleotides, window_size=window_size)

        # Compress sequence directly.
        # Header is stored inside compressed blob.

        start = time.time()

        blob, segments = compress_nucleotides_with_trained_rl(
            current_nucleotides=current_nucleotides,
            line_width=line_width,
            window_features=window_features,
            model_path=str(rl_model),
            segment_lengths=segment_lengths,
            level=9,
            parallel=parallel,
            workers=workers,
            window_size=window_size,
            device=device,
        )

        compressed_time = time.time() - start

        output_path.write_bytes(MAGIC + blob)

        print(f"[segments] {segments}")
        print(f"[num segments] {len(segments)}")
        print(f"[compressed size] {output_path.stat().st_size:,} bytes")
        print(f"[compression time] {compressed_time}s")

        # Compression metrics vs original FASTA file
        metrics = _compute_compression_metrics_vs_file(current_nucleotides=current_nucleotides, output_path=output_path)

        return


def run_decompression_pipeline(
        input_path: Path,  # path of the compressed subfile
        parallel: bool = False,
        workers: int | None = None,
) -> bytes:

    compressed_container_dir = input_path

    if not compressed_container_dir.exists():
        raise FileNotFoundError(f"Compressed folder not found: {compressed_container_dir}")

    print("\n==============================")
    print("Running decompression pipeline")
    print("==============================")

    blob = input_path.read_bytes()
    start = time.time()
    fasta_record_bytes = decompress_bytes_blocks(blob, parallel=parallel, workers=workers,)
    decompression_time = time.time() - start
    print(f"[decompression time] {decompression_time}s")

    return fasta_record_bytes



def main() -> bytes | None:
    parser = argparse.ArgumentParser(
        description="Run full RL-based multi-FASTA compression/decompression pipeline."
    )

    parser.add_argument("mode", choices=["c", "compress", "d", "decompress"], help="Use 'c' for compression or 'd' for decompression.",)
    parser.add_argument("input")
    parser.add_argument("output_path")
    parser.add_argument("--rl-model", default=DEFAULT_RL_MODEL, help=f"Path to pretrained RL model. Default: {DEFAULT_RL_MODEL}",)

    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help=f"Window size for feature extraction. Default: {DEFAULT_WINDOW_SIZE}",)

    parser.add_argument(
        "--segment-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEGMENT_LENGTHS,
        help=(
            "Segment lengths used by the RL agent. "
            f"Default: {' '.join(map(str, DEFAULT_SEGMENT_LENGTHS))}"
        ),
    )

    parser.add_argument("--parallel", action="store_true", help="Use parallel compression/decompression for blocks.",)
    parser.add_argument("--workers", type=int, default=None, help="Number of workers for parallel mode.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sequence_* files in the output folder.",)

    args = parser.parse_args()

    if args.mode in ["c", "compress"]:
        start = time.time()
        run_compression_pipeline(
                mode=args.mode,
                current_nucleotides=args.input,
                output_path=args.output_path,
                rl_model=args.rl_model,
                window_size=args.window_size,
                segment_lengths=args.segment_lengths,
                parallel=args.parallel,
                workers=args.workers,
                device=args.device,
                overwrite=args.overwrite,
        )
        end = time.time() - start
        print(f"[full compression time] {end}s")
        return None

    elif args.mode in ["d","decompress"] :
        start = time.time()
        decompressed_bytes = run_decompression_pipeline(mode=args.mode, input_path=args.input, parallel=args.parallel, workers=args.workers,)
        end = time.time() - start
        print(f"[full decompression time] {end}s")
        return decompressed_bytes

    else:
        print("Incorrect mode !")
        return None


if __name__ == "__main__":
    main()
