"""Minimal VMD (Vocaloid Motion Data 0002) writer for morph animation.

Only the morph section is written; bone/camera/light sections are emitted as
empty counts so MMD and mmd_tools can load the file normally.  Morph names are
Shift-JIS encoded, which is what MMD expects for Japanese expression names.
"""

from __future__ import annotations

from pathlib import Path
import struct

SIGNATURE = b"Vocaloid Motion Data 0002"
MODEL_NAME_BYTES = 20
MORPH_NAME_BYTES = 15


def encode_name(text: str, size: int) -> bytes:
    raw = text.encode("cp932", errors="replace")[:size]
    return raw.ljust(size, b"\x00")


def simplify(points: list[tuple[int, float]], epsilon: float = 0.015) -> list[tuple[int, float]]:
    """Keep only the keyframes that a linear interpolation cannot reproduce."""

    if len(points) <= 2:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        x1, y1 = points[first]
        x2, y2 = points[last]
        span = x2 - x1
        best_index, best_error = -1, 0.0
        for index in range(first + 1, last):
            x, y = points[index]
            expected = y1 if span == 0 else y1 + (x - x1) / span * (y2 - y1)
            error = abs(y - expected)
            if error > best_error:
                best_index, best_error = index, error
        if best_error > epsilon and best_index > 0:
            keep[best_index] = True
            stack.append((first, best_index))
            stack.append((best_index, last))
    return [point for point, flag in zip(points, keep) if flag]


def write_vmd(
    path: str | Path,
    curves: dict[str, list[tuple[int, float]]],
    *,
    model_name: str = "",
) -> int:
    """Write morph curves to ``path``; returns the number of morph keys."""

    total = sum(len(points) for points in curves.values())
    with Path(path).open("wb") as handle:
        handle.write(SIGNATURE.ljust(30, b"\x00"))
        handle.write(encode_name(model_name, MODEL_NAME_BYTES))
        handle.write(struct.pack("<I", 0))  # bone frames
        handle.write(struct.pack("<I", total))  # morph frames
        for name, points in curves.items():
            encoded = encode_name(name, MORPH_NAME_BYTES)
            for frame, weight in points:
                handle.write(encoded)
                handle.write(struct.pack("<I", max(0, int(round(frame)))))
                handle.write(struct.pack("<f", float(weight)))
        for _section in range(4):  # camera, light, self shadow, IK
            handle.write(struct.pack("<I", 0))
    return total


def read_vmd_morphs(path: str | Path) -> dict[str, int]:
    """Read back morph names of a VMD file; used by the self-check."""

    data = Path(path).read_bytes()
    if not data.startswith(SIGNATURE):
        raise ValueError("not a VMD 0002 file")
    offset = 30 + MODEL_NAME_BYTES
    (bone_count,) = struct.unpack_from("<I", data, offset)
    offset += 4 + bone_count * 111
    (morph_count,) = struct.unpack_from("<I", data, offset)
    offset += 4
    curves: dict[str, int] = {}
    for _ in range(morph_count):
        raw_name, _frame, _weight = struct.unpack_from("<15sIf", data, offset)
        offset += 23
        name = raw_name.split(b"\x00")[0].decode("cp932", errors="replace")
        curves[name] = curves.get(name, 0) + 1
    return curves


__all__ = ["encode_name", "simplify", "write_vmd", "read_vmd_morphs"]
