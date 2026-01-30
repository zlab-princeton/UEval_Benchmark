# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import torch

import importlib as imp
import os.path as osp

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from src.utils.model_utils import build_emu3p5_vllm
from src.utils.vllm_generation_utils import generate
from src.utils.generation_utils import multimodal_decode
from src.utils.painting_utils import ProtoWriter
from src.utils.input_utils import build_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="", type=str)
    parser.add_argument("--tensor-parallel-size", default=4, type=int)
    parser.add_argument("--gpu-memory-utilization", default=0.7, type=float)
    parser.add_argument("--seed", default=6666, type=int)
    parser.add_argument("--file-path", default=None, type=str,
                        help="JSON file containing prompts.")
    parser.add_argument("--dataset-name", default="primerL/UEval-all",
                        type=str, help="Hugging Face dataset name.")
    parser.add_argument("--dataset-split", default=None, type=str,
                        help="Dataset split name (e.g., 'art', 'code', etc.).")
    args = parser.parse_args()
    return args


def _normalize_entries(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "entries", "examples", "prompts"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return []


def _load_prompts_from_file(file_path):
    if not file_path:
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    entries = _normalize_entries(payload)
    prompts = []
    seen_ids = set()
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            continue
        prompt_text = item.get("prompt") or item.get(
            "question") + "please generated images and caption."
        if not prompt_text:
            continue
        ref_image = None

        if ref_image is None:
            question = prompt_text
        else:
            question = {
                "prompt": prompt_text,
                "reference_image": ref_image,
            }
        name = item.get("id")
        if name is None:
            name = f"{idx:03d}"
        else:
            name = str(name)
        if name in seen_ids:
            continue
        seen_ids.add(name)
        prompts.append((name, question))
    return prompts


def _load_prompts_from_hf_dataset(dataset_name, split_name=None):
    """
    Load prompts from a Hugging Face dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "primerL/UEval-all")
        split_name: Specific split to load (e.g., "art", "code", etc.)

    Returns:
        List of (id, question) tuples
    """
    if not dataset_name:
        return None

    print(f"[INFO] Loading dataset {dataset_name}", flush=True)

    try:
        if split_name:
            dataset = load_dataset(dataset_name, split=split_name)
        else:
            dataset = load_dataset(dataset_name)
            # If dataset has multiple splits, use the first one
            if isinstance(dataset, dict):
                split_name = list(dataset.keys())[0]
                dataset = dataset[split_name]
                print(f"[INFO] Using split: {split_name}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}", flush=True)
        return None

    prompts = []
    seen_ids = set()

    for idx, item in enumerate(dataset):
        if not isinstance(item, dict):
            continue

        # Get prompt text
        prompt_text = item.get("prompt") or item.get("question")
        if not prompt_text:
            continue

        # For now, we don't use reference images from the dataset
        # but the structure supports it if needed in the future
        ref_image = None

        if ref_image is None:
            question = prompt_text
        else:
            question = {
                "prompt": prompt_text,
                "reference_image": ref_image,
            }

        # Get ID
        name = item.get("id")
        if name is None:
            name = f"{idx:03d}"
        else:
            name = str(name)

        if name in seen_ids:
            continue

        seen_ids.add(name)
        prompts.append((name, question))

    print(f"[INFO] Loaded {len(prompts)} prompts from dataset", flush=True)
    return prompts


def _ensure_named_prompts(sequence):
    if sequence is None:
        return []
    if isinstance(sequence, dict):
        return [(str(name), payload) for name, payload in sequence.items()]
    named_prompts = []
    for idx, entry in enumerate(sequence):
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            maybe_name, maybe_payload = entry
            if isinstance(maybe_name, (str, int)):
                named_prompts.append((str(maybe_name), maybe_payload))
                continue
        named_prompts.append((f"{idx+1:03d}", entry))
    return named_prompts


def inference(
    cfg,
    model,
    tokenizer,
    vq_model,
    proto_dir=None,
):
    save_path = Path(cfg.save_path)
    proto_dir = Path(proto_dir) if proto_dir else save_path / "proto"

    save_path.mkdir(parents=True, exist_ok=True)
    proto_dir.mkdir(parents=True, exist_ok=True)
    proto_writer = ProtoWriter()

    for name, question in tqdm(cfg.prompts, total=len(cfg.prompts)):
        proto_file = proto_dir / f"{name}.pb"
        if proto_file.exists():
            print(
                f"[WARNING] Result already exists, skipping {name}", flush=True)
            continue

        torch.cuda.empty_cache()

        reference_image = None
        if not isinstance(question, str):
            if isinstance(question["reference_image"], list):
                print(
                    f"[INFO] {len(question['reference_image'])} reference images are provided")
                reference_image = []
                for img in question["reference_image"]:
                    reference_image.append(Image.open(img).convert("RGB"))
            else:
                print(f"[INFO] 1 reference image is provided")
                reference_image = Image.open(
                    question["reference_image"]).convert("RGB")
            question = question["prompt"]
        else:
            print(f"[INFO] No reference image is provided")

        proto_writer.clear()
        proto_writer.extend([["question", question]])
        if reference_image is not None:
            if isinstance(reference_image, list):
                for idx, img in enumerate(reference_image):
                    proto_writer.extend([[f"reference_image", img]])
            else:
                proto_writer.extend([["reference_image", reference_image]])

        success = True
        prompt = cfg.template.format(question=question)

        print(f"[INFO] Handling prompt: {prompt}")
        if reference_image is not None:
            if isinstance(reference_image, list):
                image_str = ""
                for img in reference_image:
                    image_str += build_image(img, cfg, tokenizer, vq_model)
            else:
                image_str = build_image(
                    reference_image, cfg, tokenizer, vq_model)
            prompt = prompt.replace("<|IMAGE|>", image_str)
            unc_prompt = cfg.unc_prompt.replace("<|IMAGE|>", image_str)
        else:
            unc_prompt = cfg.unc_prompt

        input_ids = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False)

        if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
            BOS = torch.Tensor(
                [[cfg.special_token_ids["BOS"]]], dtype=input_ids.dtype)
            input_ids = torch.cat([BOS, input_ids], dim=1)

        unconditional_ids = tokenizer.encode(
            unc_prompt, return_tensors="pt", add_special_tokens=False)

        for result_tokens in generate(cfg, model, tokenizer, input_ids, unconditional_ids):
            try:
                print(f"{result_tokens.shape=}")
                result = tokenizer.decode(
                    result_tokens, skip_special_tokens=False)
                mm_out = multimodal_decode(result, tokenizer, vq_model)
                proto_writer.extend(mm_out)
            except Exception as e:
                success = False
                print(f"[ERROR] Failed to generate token sequence: {e}")
                break

        if not success:
            continue

        proto_writer.save(str(proto_file))


def main():
    args = parse_args()
    cfg_name = Path(args.cfg).stem
    cfg_package = Path(args.cfg).parent.__str__().replace("/", ".")
    cfg = imp.import_module(f".{cfg_name}", package=cfg_package)

    # Priority: file_path > dataset > cfg.prompts
    prompts_from_file = _load_prompts_from_file(args.file_path)
    if prompts_from_file is not None:
        cfg.prompts = prompts_from_file
    elif args.dataset_name:
        prompts_from_dataset = _load_prompts_from_hf_dataset(
            args.dataset_name, args.dataset_split)
        if prompts_from_dataset is not None:
            cfg.prompts = prompts_from_dataset

    cfg.prompts = _ensure_named_prompts(cfg.prompts)

    # Determine proto directory name
    if args.file_path:
        proto_dir_name = Path(args.file_path).stem
    elif args.dataset_split:
        proto_dir_name = args.dataset_split
    else:
        proto_dir_name = "proto"

    proto_dir = Path(cfg.save_path) / proto_dir_name

    cfg.prompts = [(n, p) for n, p in cfg.prompts if not (
        proto_dir / f"{n}.pb").exists()]
    cfg.num_prompts = len(cfg.prompts)

    model, tokenizer, vq_model = build_emu3p5_vllm(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        vq_device=cfg.vq_device,
        seed=cfg.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **getattr(cfg, "diffusion_decoder_kwargs", {}),
    )
    print(f"[INFO] Model loaded successfully")
    print(f"[Debug]{cfg.prompts}")
    cfg.special_token_ids = {}
    for k, v in cfg.special_tokens.items():
        cfg.special_token_ids[k] = tokenizer.encode(v)[0]

    inference(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        vq_model=vq_model,
        proto_dir=proto_dir,
    )
    print(f"[INFO] Inference finished")


if __name__ == "__main__":
    main()
