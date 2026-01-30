#!/usr/bin/env python3
"""
Parse EMU protobuf files (e.g., the ``art`` split) and generate JSON output.

This script processes protobuf files from a directory, extracts images and text,
and generates a JSON file with id, emu_image, and emu_text fields.

python vis_proto_ueval.py \
        --proto-dir Emu3.5/outputs \
        --image-dir Emu3.5/image \
        --output-json output.json

"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, List, Tuple

from google.protobuf.message import DecodeError

from vis_proto import parse_story


def _add_chunk(chunks: List[str], label: str, content: str | None) -> None:
    if not content:
        return
    text = content.strip()
    if not text:
        return
    prefix = f"[{label}] " if label else ""
    chunks.append(f"{prefix}{text}")


def _ensure_relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _extract_images_and_text(
    story: Any,
    image_dir: Path,
    relative_root: Path,
) -> Tuple[List[str], str]:
    image_dir.mkdir(parents=True, exist_ok=True)

    saved_images: List[str] = []
    text_chunks: List[str] = []

    _add_chunk(text_chunks, "global_cot", getattr(story, "summary", ""))

    for ref_idx, ref_image in enumerate(getattr(story, "reference_images", [])):
        image_container = getattr(ref_image, "image", None)
        image_bytes = getattr(image_container, "image_data", None)
        if image_bytes:
            file_path = image_dir / f"ref_{ref_idx:02d}.png"
            file_path.write_bytes(image_bytes)
            saved_images.append(_ensure_relative(file_path, relative_root))
        _add_chunk(
            text_chunks,
            f"reference_{ref_idx:02d}_cot",
            getattr(ref_image, "chain_of_thought", ""),
        )

    for clip_idx, clip in enumerate(getattr(story, "clips", [])):
        clip_name = (
            getattr(clip, "clip_name", None)
            or getattr(clip, "name", None)
            or f"clip_{clip_idx:02d}"
        )
        for seg_idx, segment in enumerate(getattr(clip, "segments", [])):
            seg_label = f"{clip_name}_seg_{seg_idx:02d}"
            _add_chunk(text_chunks, seg_label, getattr(segment, "asr", ""))
            for img_idx, image in enumerate(getattr(segment, "images", [])):
                image_container = getattr(image, "image", None)
                image_bytes = getattr(image_container, "image_data", None)
                if image_bytes:
                    file_path = image_dir / \
                        f"{clip_name}_{seg_idx:02d}_{img_idx:02d}.png"
                    file_path.write_bytes(image_bytes)
                    saved_images.append(
                        _ensure_relative(file_path, relative_root))
                _add_chunk(
                    text_chunks,
                    f"{seg_label}_img_{img_idx:02d}_cot",
                    getattr(image, "chain_of_thought", ""),
                )

    return saved_images, "\n".join(text_chunks)


def process_art_data(
    proto_dir: Path,
    output_json: Path,
    image_root: Path,
    relative_root: Path,
) -> None:
    data: List[dict[str, Any]] = []

    for proto_path in sorted(proto_dir.glob("*.pb")):
        proto_id = proto_path.stem

        try:
            story = parse_story(str(proto_path))
        except DecodeError as exc:
            logging.error(
                "Failed to parse %s: %s. Deleting file.", proto_path, exc)
            try:
                proto_path.unlink()
                logging.warning("Deleted unreadable protobuf: %s", proto_path)
            except OSError as delete_err:
                logging.warning("Unable to delete %s: %s",
                                proto_path, delete_err)
            continue

        image_dir = image_root / proto_path.stem
        image_paths, text_blob = _extract_images_and_text(
            story,
            image_dir=image_dir,
            relative_root=relative_root,
        )

        entry = {
            "id": proto_id,
            "emu_image": image_paths,
            "emu_text": text_blob
        }
        data.append(entry)

    with output_json.open("w", encoding="utf-8") as dst:
        json.dump(data, dst, ensure_ascii=False, indent=2)

    logging.info("Processed %d protobuf files and saved to %s",
                 len(data), output_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate JSON from EMU protobuf files with id, emu_image, and emu_text fields."
    )
    parser.add_argument(
        "--proto-dir",
        required=True,
        help="Directory that stores the .pb files.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to the output JSON file to be created.",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Base directory used to store the decoded EMU images.",
    )
    parser.add_argument(
        "--relative-root",
        default=".",
        help="Base directory for computing relative image paths (default: current directory).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    proto_dir = Path(args.proto_dir)
    output_json = Path(args.output_json)
    image_root = Path(args.image_dir).resolve()
    relative_root = Path(args.relative_root).resolve()

    process_art_data(
        proto_dir=proto_dir,
        output_json=output_json,
        image_root=image_root,
        relative_root=relative_root,
    )


if __name__ == "__main__":
    main()
