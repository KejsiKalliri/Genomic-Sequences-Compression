from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from C_core import compress_block, decompress_block
from pretrain_model import DQNAgentTorch
from RL import (build_state, load_feature_cache, rollout_segmentation)


MAGIC = b"RLC"
VERSION = 2

MODE_RAW = 0
MODE_FASTA = 1
MODE_SEQUENCE = 2  # raw nucleotide sequence + line_width metadata


# ---------------------------- Binary helpers ----------------------------

def pack_u32(x: int) -> bytes:
    return x.to_bytes(4, "little")


def unpack_u32(data: bytes, off: int) -> tuple[int, int]:
    if off + 4 > len(data):
        raise ValueError("truncated u32")
    return int.from_bytes(data[off:off + 4], "little"), off + 4


def pack_bytes(data: bytes) -> bytes:
    return pack_u32(len(data)) + data


def unpack_bytes(data: bytes, off: int) -> tuple[bytes, int]:
    n, off = unpack_u32(data, off)
    if off + n > len(data):
        raise ValueError("truncated metadata")
    return data[off:off + n], off + n


def read_fasta_with_format(path: str | Path) -> tuple[str, str, int]:
    header = ""
    seq_parts: list[str] = []
    line_width = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue

            if line.startswith(">"):
                if not header:
                    header = line
                continue

            if line_width == 0:
                line_width = len(line)

            seq_parts.append(line.upper())

    sequence = "".join(seq_parts)

    if not sequence:
        raise ValueError("No sequence found in FASTA file")

    if not header:
        header = ">sequence"

    if line_width <= 0:
        line_width = 60

    return header, sequence, line_width


def format_fasta_bytes(sequence: bytes, header: str, line_width: int) -> bytes:
    if line_width <= 0:
        line_width = 60

    out = bytearray()
    out += header.encode("ascii") + b"\n"

    for i in range(0, len(sequence), line_width):
        out += sequence[i:i + line_width] + b"\n"

    return bytes(out)


# ---------------------------- Fixed block mode ----------------------------

def split_blocks_fixed(data: bytes, block_size: int) -> list[bytes]:
    return [data[i:i + block_size] for i in range(0, len(data), block_size)]


def _compress_one(args: tuple[bytes, int]) -> bytes:
    block, level = args
    return compress_block(block, level=level)


def _decompress_one(args: tuple[bytes, int]) -> bytes:
    block, raw_len = args
    return decompress_block(block, expected_output_len=raw_len)


def compress_blocks_to_blob(
    blocks: list[bytes],
    level: int = 9,
    parallel: bool = False,
    workers: int | None = None,
    mode: int = MODE_RAW,
    line_width: int = 60,
    fasta_header: str = ">sequence",
) -> bytes:
    """
    Compresses a list of raw blocks into a binary blob.

    Binary layout, VERSION 2:
        MAGIC
        VERSION
        MODE

        if MODE_FASTA:
            line_width:uint32
            fasta_header_length:uint32
            fasta_header bytes

        if MODE_SEQUENCE:
            line_width:uint32

        num_blocks:uint32
        repeated blocks:
            raw_length:uint32
            compressed_length:uint32
            compressed_payload bytes
    """

    if line_width <= 0:
        line_width = 60

    if mode not in {MODE_RAW, MODE_FASTA, MODE_SEQUENCE}:
        raise ValueError(f"bad mode: {mode}")

    if parallel:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            compressed_blocks = list(ex.map(_compress_one, [(b, level) for b in blocks]))
    else:
        compressed_blocks = [compress_block(b, level=level) for b in blocks]

    out = bytearray()
    out += MAGIC
    out += bytes([VERSION])
    out += bytes([mode])

    if mode == MODE_FASTA:
        if not fasta_header:
            fasta_header = ">sequence"
        if not fasta_header.startswith(">"):
            fasta_header = ">" + fasta_header

        out += pack_u32(line_width)
        out += pack_bytes(fasta_header.encode("ascii"))

    elif mode == MODE_SEQUENCE:
        out += pack_u32(line_width)

    out += pack_u32(len(blocks))

    for raw_block, comp_block in zip(blocks, compressed_blocks):
        out += pack_u32(len(raw_block))
        out += pack_u32(len(comp_block))
        out += comp_block

    return bytes(out)


def compress_bytes_blocks(data: bytes, block_size: int, level: int = 9, parallel: bool = False, workers: int | None = None) -> bytes:
    blocks = split_blocks_fixed(data, block_size)
    return compress_blocks_to_blob(blocks, level=level, parallel=parallel, workers=workers, mode=MODE_RAW)


def decompress_bytes_blocks_with_metadata(
    blob: bytes,
    parallel: bool = False,
    workers: int | None = None,
) -> tuple[bytes, int, int]:
    """
    Decompresses a blob and returns:
        decompressed_bytes, line_width, mode

    For MODE_SEQUENCE, line_width is read from the compressed file.
    For MODE_RAW, line_width defaults to 60.
    For MODE_FASTA, the returned bytes are a formatted FASTA record.
    """

    off = 0

    if blob[:3] != MAGIC:
        raise ValueError("bad magic")
    off += 3

    if off >= len(blob):
        raise ValueError("truncated version")

    version = blob[off]
    off += 1

    mode = MODE_RAW
    fasta_header = ">sequence"
    line_width = 60

    if version == 1:
        # Old format:
        # MAGIC + VERSION + num_blocks + blocks...
        mode = MODE_RAW

    elif version == 2:
        # New format:
        # MAGIC + VERSION + MODE + optional metadata + num_blocks + blocks...
        if off >= len(blob):
            raise ValueError("truncated mode")

        mode = blob[off]
        off += 1

        if mode == MODE_FASTA:
            line_width, off = unpack_u32(blob, off)
            header_bytes, off = unpack_bytes(blob, off)
            fasta_header = header_bytes.decode("ascii")

        elif mode == MODE_SEQUENCE:
            line_width, off = unpack_u32(blob, off)

        elif mode == MODE_RAW:
            pass

        else:
            raise ValueError(f"bad mode: {mode}")

    else:
        raise ValueError("bad version")

    num_blocks, off = unpack_u32(blob, off)

    blocks: list[bytes] = []
    raw_lengths: list[int] = []

    for _ in range(num_blocks):
        raw_len, off = unpack_u32(blob, off)
        comp_len, off = unpack_u32(blob, off)

        if off + comp_len > len(blob):
            raise ValueError(
                f"truncated block payload: off={off}, "
                f"comp_len={comp_len}, file_size={len(blob)}"
            )

        comp = blob[off:off + comp_len]
        off += comp_len

        blocks.append(comp)
        raw_lengths.append(raw_len)

    if off != len(blob):
        raise ValueError(f"extra bytes after payload: off={off}, file_size={len(blob)}")

    if parallel:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            dec_blocks = list(ex.map(_decompress_one, zip(blocks, raw_lengths)))
    else:
        dec_blocks = [
            decompress_block(comp, expected_output_len=raw_len)
            for comp, raw_len in zip(blocks, raw_lengths)
        ]

    out = bytearray()

    for raw_len, dec in zip(raw_lengths, dec_blocks):
        if len(dec) != raw_len:
            raise ValueError(f"raw_len mismatch: expected {raw_len}, got {len(dec)}")

        out += dec

    if mode == MODE_FASTA:
        return format_fasta_bytes(bytes(out), fasta_header, line_width), line_width, mode

    return bytes(out), line_width, mode


def decompress_bytes_blocks(
    blob: bytes,
    parallel: bool = False,
    workers: int | None = None,
) -> bytes:
    """
    Backward-compatible decompression function.
    It returns only the decompressed bytes.
    Use decompress_bytes_blocks_with_metadata() when you also need line_width.
    """

    decompressed_bytes, _line_width, _mode = decompress_bytes_blocks_with_metadata(
        blob,
        parallel=parallel,
        workers=workers,
    )

    return decompressed_bytes


def decompress_nucleotides_with_metadata(
    blob: bytes,
    parallel: bool = False,
    workers: int | None = None,
) -> tuple[bytes, int]:
    """
    Decompresses a sequence-level blob.

    Returns:
        nucleotide_sequence_bytes, line_width
    """

    decompressed_bytes, line_width, mode = decompress_bytes_blocks_with_metadata(
        blob,
        parallel=parallel,
        workers=workers,
    )

    if mode == MODE_FASTA:
        raise ValueError(
            "Expected a sequence-level blob, but got MODE_FASTA. "
            "Use decompress_bytes_blocks() for FASTA-formatted output."
        )

    return decompressed_bytes, line_width


# ---------------------------- RL segment-length mode ----------------------------


def infer_state_dim(window_features: list[list[float]]) -> int:

    dummy_current = window_features[0]

    dummy_next1 = window_features[1] if len(window_features) > 1 else None  # features of window 1
    dummy_next2 = window_features[2] if len(window_features) > 2 else None  # features of window 2
    dummy_next3 = window_features[3] if len(window_features) > 3 else None  # features of window 3

    return len(build_state(dummy_current, dummy_next1, dummy_next2, dummy_next3,0.0,1.0))


def compress_fasta_with_trained_seglen_rl(
    fasta_path: str,
    cache_path: str,
    model_path: str,   # pretrained model
    segment_lengths: list[int],
    level: int = 9,
    parallel: bool = False,
    workers: int | None = None,
    window_size: int = 1000,
    device: str | None = None,
) -> tuple[bytes, list[tuple[int, int]]]:

    fasta_header, sequence, line_width = read_fasta_with_format(fasta_path)
    cache = load_feature_cache(cache_path)  # features of every window of the sequence

    if cache["window_size"] != window_size:
        raise ValueError("Window size mismatch between cache and CLI")

    window_features = cache["window_features"]
    state_dim = infer_state_dim(window_features)

    agent = DQNAgentTorch.load(model_path, state_dim=state_dim, action_dim=len(segment_lengths), device=device,)

    segments = rollout_segmentation(
        sequence=sequence,
        window_features=window_features,
        agent=agent,
        segment_lengths=segment_lengths,
        window_size=window_size,
        level=level,
    )

    blocks = [sequence[start:end].encode("ascii") for start, end in segments]  # return every segment in bytes

    blob = compress_blocks_to_blob(
        blocks,
        level=level,
        parallel=parallel,
        workers=workers,
        mode=MODE_FASTA,
        line_width=line_width,
        fasta_header=fasta_header,
    )

    return blob, segments

def compress_nucleotides_with_trained_rl(
    current_nucleotides : str,
    line_width : int,
    window_features: list[list[float]],
    model_path: str,   # pretrained model
    segment_lengths: list[int],
    level: int = 9,
    parallel: bool = False,
    workers: int | None = None,
    window_size: int = 1000,
    device: str | None = None,
) -> tuple[bytes, list[tuple[int, int]]]:


    state_dim = infer_state_dim(window_features)

    agent = DQNAgentTorch.load(model_path, state_dim=state_dim, action_dim=len(segment_lengths), device=device,)

    segments = rollout_segmentation(
        sequence=current_nucleotides,
        window_features=window_features,
        agent=agent,
        segment_lengths=segment_lengths,
        window_size=window_size,
        level=level,
    )

    blocks = [current_nucleotides[start:end].encode("ascii") for start, end in segments]

    if line_width <= 0:
        line_width = 60

    blob = compress_blocks_to_blob(
        blocks,
        level=level,
        parallel=parallel,
        workers=workers,
        mode=MODE_SEQUENCE,
        line_width=line_width,
    )

    return blob, segments


def compress_sequence_with_header_trained_seglen_rl(
    sequence: str,
    fasta_header: str,
    line_width: int,
    cache_path: str,
    model_path: str,
    segment_lengths: list[int],
    level: int = 9,
    parallel: bool = False,
    workers: int | None = None,
    window_size: int = 10000,
    device: str | None = None,
) -> tuple[bytes, list[tuple[int, int]]]:
    """
    Compresses one DNA sequence directly from memory.

    Important:
    - No temporary FASTA file is created.
    - The FASTA header is still stored inside the compressed blob.
    - The blob uses MODE_FASTA, so decompression reconstructs the FASTA record.
    """

    sequence = sequence.strip().upper()

    if not sequence:
        raise ValueError("Empty sequence received for compression.")

    if not fasta_header:
        fasta_header = ">sequence"

    if not fasta_header.startswith(">"):
        fasta_header = ">" + fasta_header

    if line_width <= 0:
        line_width = 60

    cache = load_feature_cache(cache_path)

    if cache["window_size"] != window_size:
        raise ValueError(
            f"Window size mismatch: cache has {cache['window_size']}, "
            f"but pipeline uses {window_size}."
        )

    if cache.get("sequence_length") is not None:
        if cache["sequence_length"] != len(sequence):
            raise ValueError(
                f"Sequence length mismatch: cache has {cache['sequence_length']}, "
                f"but sequence has {len(sequence)}."
            )

    window_features = cache["window_features"]

    state_dim = infer_state_dim(window_features)

    agent = DQNAgentTorch.load(model_path, state_dim=state_dim, action_dim=len(segment_lengths), device=device,)

    segments = rollout_segmentation(
        sequence=sequence,
        window_features=window_features,
        agent=agent,
        segment_lengths=segment_lengths,
        window_size=window_size,
        level=level,
    )

    blocks = [sequence[start:end].encode("ascii") for start, end in segments]

    blob = compress_blocks_to_blob(
        blocks,
        level=level,
        parallel=parallel,
        workers=workers,
        mode=MODE_FASTA,
        line_width=line_width,
        fasta_header=fasta_header,
    )

    return blob, segments


# ---------------------------- CLI ----------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Compression runner for fixed blocks or RL segment-length model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_c = sub.add_parser("c", help="compress")
    ap_c.add_argument("input", type=str)
    ap_c.add_argument("output", type=str)
    # ap_c.add_argument("--block-size", type=int, default=900_000)
    block_size = 900_000
    # ap_c.add_argument("--level", type=int, default=9)
    level = 9
    ap_c.add_argument("--parallel", action="store_true")
    ap_c.add_argument("--workers", type=int, default=None)

    ap_c.add_argument("--rl-model", type=str, default=None)
    ap_c.add_argument("--feature-cache", type=str, default=None)
    ap_c.add_argument("--window-size", type=int, default=1000)
    ap_c.add_argument("--segment-lengths", type=int, nargs="+", default=[100000, 200000, 400000, 800000])
    ap_c.add_argument("--device", type=str, default=None)

    ap_d = sub.add_parser("d", help="decompress")
    ap_d.add_argument("input", type=str)
    ap_d.add_argument("output", type=str)
    ap_d.add_argument("--parallel", action="store_true")
    ap_d.add_argument("--workers", type=int, default=None)

    args = ap.parse_args(argv[1:])

    try:
        if args.cmd == "c":
            t0 = time.time()

            if args.rl_model is not None:
                if args.feature_cache is None:
                    raise ValueError("--feature-cache is required when using --rl-model")

                blob, segments = compress_fasta_with_trained_seglen_rl(
                    fasta_path=args.input,
                    cache_path=args.feature_cache,
                    model_path=args.rl_model,
                    segment_lengths=args.segment_lengths,
                    level=level,  # args.level
                    parallel=args.parallel,
                    workers=args.workers,
                    window_size=args.window_size,
                    device=args.device,
                )
                Path(args.output).write_bytes(blob)
                print(f"RL segments: {segments}")
                print(f"Num segments: {len(segments)}")

            else:
                data = Path(args.input).read_bytes()
                blob = compress_bytes_blocks(
                    data,
                    block_size=block_size, # args.block_size
                    level=level,
                    parallel=args.parallel,
                    workers=args.workers,
                )
                Path(args.output).write_bytes(blob)

            print(f"Compressed in {time.time() - t0:.2f}s")
            return 0

        if args.cmd == "d":
            t0 = time.time()
            blob = Path(args.input).read_bytes()
            raw = decompress_bytes_blocks(
                blob,
                parallel=args.parallel,
                workers=args.workers,
            )
            Path(args.output).write_bytes(raw)
            print(f"Decompressed in {time.time() - t0:.2f}s")
            return 0

        return 1

    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

