#!/usr/bin/env python3
"""
Generate text and image outputs using Gemini's multimodal generation API.
Loads prompts from HuggingFace dataset (primerL/UEval-all).

This script reads prompts from HuggingFace and generates both text and image responses
using Google's Gemini API with image generation capabilities.

Setup:
    1. Install required packages:
       pip install google-genai pillow datasets

    2. Set your Gemini API key:
       export GEMINI_API_KEY="your-api-key-here"

Usage:
    # Generate for all domains
    python gemini.py \
        --api_key AIzaSyB63m9zejxoGVyzXDynTBakfZtPVzXmuSM\
        --output_path Gemini2.5/test.json \
        --output_image_dir Gemini2.5/test/ \
        --limit 1

    # Generate for specific domains
    python gemini.py \
        --output_path Gemini2.0/art_results.json \
        --output_image_dir Gemini2.0/images/ \
        --domains art life tech

    # Limit number of items
    python gemini.py \
        --output_path Gemini2.0/test.json \
        --output_image_dir Gemini2.0/images/ \
        --limit 10

Output JSON format:
    [
        {
            "id": 1,
            "prompt": "Your prompt here...",
            "task_type": "art",
            "question_type": "open",
            "gemini_image_ans": ["Gemini2.0/images/1_1.png"],
            "gemini_text_ans": "Generated text response..."
        },
        ...
    ]
"""

import argparse
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from google import genai
from google.genai import types

try:
    from datasets import load_dataset
except ImportError:
    print("Warning: 'datasets' library not found. Install with: pip install datasets")
    load_dataset = None


class GeminiMultimodalGenerator:
    """Generator for text and image content using Gemini API."""

    def __init__(self, model_name: str, api_key: str):
        """
        Initialize the Gemini generator.

        Args:
            model_name: Name of the Gemini model to use
            api_key: Gemini API key
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(
        self,
        prompt: Dict[str, Any],
        output_image_dir: str,
        retry_delay: float = 3.0,
        max_attempts: int = 100,
    ) -> Tuple[List[str], str]:
        """
        Generate text and images from a prompt.

        Args:
            prompt: Dictionary containing 'id' and 'prompt' fields
            output_image_dir: Directory to save generated images
            retry_delay: Seconds to wait between retry attempts
            max_attempts: Maximum number of retry attempts (None for infinite)

        Returns:
            Tuple of (image_paths, text_response)

        Raises:
            RuntimeError: If max_attempts is reached without success
        """
        prompt_text = prompt["prompt"]
        prompt_id = prompt["id"]
        attempt = 0

        while max_attempts is None or attempt < max_attempts:
            attempt += 1
            try:
                print(
                    f"Attempt {attempt}: Generating content for ID {prompt_id}...")

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"]
                    ),
                )

                # Validate response structure
                if (
                    not response
                    or not response.candidates
                    or not response.candidates[0].content.parts
                ):
                    print(
                        f"Attempt {attempt} failed: Invalid response, retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    continue

                output_text = ""
                output_paths = []
                image_counter = 1

                # Process response parts
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        output_text += part.text
                        print(f"Generated text (length: {len(part.text)})")

                    if part.inline_data is not None:
                        image = Image.open(BytesIO(part.inline_data.data))
                        safe_filename = f"{prompt_id}_{image_counter}.png"
                        os.makedirs(output_image_dir, exist_ok=True)
                        image_path = os.path.join(
                            output_image_dir, safe_filename)
                        image.save(image_path)
                        output_paths.append(image_path)
                        print(f"Saved image: {image_path}")
                        image_counter += 1

                # Validate that both text and images were generated
                if output_text and output_paths:
                    print(f"Attempt {attempt} succeeded!")
                    print(f"Generated text length: {len(output_text)}")
                    print(f"Generated images: {len(output_paths)}")
                    return output_paths, output_text
                else:
                    if not output_text:
                        print(
                            f"Attempt {attempt} failed: No text generated, retrying in {retry_delay}s..."
                        )
                    if not output_paths:
                        print(
                            f"Attempt {attempt} failed: No images generated, retrying in {retry_delay}s..."
                        )
                    time.sleep(retry_delay)
                    continue

            except Exception as e:
                print(
                    f"Attempt {attempt} failed with error: {e}, retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                continue

        raise RuntimeError(
            f"Failed to generate content after {max_attempts} attempts for ID {prompt_id}"
        )


def load_dataset_from_hf(
    hf_dataset_id: str,
    domains: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load evaluation dataset from HuggingFace.

    Args:
        hf_dataset_id: HuggingFace dataset ID
        domains: List of task types to filter (e.g., ['art', 'life', 'tech'])
        limit: Maximum number of items to load

    Returns:
        List of dataset items
    """
    if load_dataset is None:
        raise RuntimeError(
            "The 'datasets' library is required to load from HuggingFace. "
            "Install with: pip install datasets"
        )

    print(f"Loading dataset from HuggingFace: {hf_dataset_id}")
    try:
        dataset = load_dataset(hf_dataset_id, split="test")
        # Convert to list of dicts
        data = [dict(item) for item in dataset]
        print(f"Loaded {len(data)} items from HuggingFace")

        # Filter by domains if specified
        if domains:
            domains_set = set(d.lower() for d in domains)
            data = [
                item for item in data
                if item.get("task_type", "").lower() in domains_set
            ]
            print(f"Filtered to {len(data)} items matching domains: {domains}")

        # Apply limit if specified
        if limit:
            data = data[:limit]
            print(f"Limited to first {len(data)} items")

        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from HuggingFace: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text and images using Gemini from HuggingFace dataset"
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="Gemini API key (if not provided, reads from GEMINI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--hf_dataset",
        default="primerL/UEval-all",
        help="HuggingFace dataset ID (default: primerL/UEval-all)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Filter by task types (e.g., --domains art life tech). "
             "Available: art, life, tech, exercise, space, textbook, diagram, paper",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save output JSON file with results",
    )
    parser.add_argument(
        "--output_image_dir",
        required=True,
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-image",
        help="Gemini model name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process (for testing)",
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=3.0,
        help="Seconds to wait between retry attempts (default: 3.0)",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=100,
        help="Maximum retry attempts per prompt (default: 100, set to 0 for infinite)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="Save checkpoint every N items (default: 1, set to 0 to disable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Get API key from argument or environment variable
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key is required. Provide it via --api_key argument "
            "or set GEMINI_API_KEY environment variable."
        )

    # Load data from HuggingFace
    print(f"\n{'='*60}")
    print("Loading dataset from HuggingFace...")
    print(f"{'='*60}")
    data = load_dataset_from_hf(
        hf_dataset_id=args.hf_dataset,
        domains=args.domains,
        limit=args.limit,
    )

    if not data:
        print("No data to process. Exiting.")
        return

    print(f"\n{'='*60}")
    print(f"Will process {len(data)} items")
    print(f"{'='*60}\n")

    # Initialize generator
    generator = GeminiMultimodalGenerator(
        model_name=args.model,
        api_key=api_key,
    )

    # Setup checkpoint file
    output_path = Path(args.output_path)
    checkpoint_path = output_path.parent / \
        f".{output_path.stem}_checkpoint.json"

    # Try to load existing checkpoint
    outputs = []
    processed_ids = set()

    if checkpoint_path.exists() and args.checkpoint_interval > 0:
        print(f"Found checkpoint file: {checkpoint_path}")
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            outputs = checkpoint_data.get("outputs", [])
            processed_ids = {item["id"] for item in outputs}
            print(
                f"Loaded {len(outputs)} generated items, continuing from item {len(outputs) + 1}\n"
            )
        except Exception as e:
            print(f"Failed to load checkpoint: {e}, starting from beginning\n")
            outputs = []
            processed_ids = set()

    # Process each prompt
    for idx, item in enumerate(data):
        item_id = item.get("id")

        # Skip if already processed
        if item_id in processed_ids:
            print(f"⏭️  Skipping already generated item ID {item_id}")
            continue

        print(f"\n{'='*60}")
        print(f"[{idx + 1}/{len(data)}] Processing ID {item_id}")
        print(f"Task: {item.get('task_type', 'N/A')}")
        print(f"Question type: {item.get('question_type', 'N/A')}")
        print(f"{'='*60}")

        try:
            max_attempts = args.max_attempts if args.max_attempts > 0 else None
            image_paths, text_response = generator.generate(
                prompt=item,
                output_image_dir=args.output_image_dir,
                retry_delay=args.retry_delay,
                max_attempts=max_attempts,
            )

            output_item = {
                "id": item_id,
                "prompt": item.get("prompt", ""),
                "task_type": item.get("task_type", ""),
                "question_type": item.get("question_type", ""),
                "gemini_image_ans": image_paths,
                "gemini_text_ans": text_response,
            }

            outputs.append(output_item)

            # Save checkpoint periodically
            if args.checkpoint_interval > 0 and (len(outputs) % args.checkpoint_interval == 0):
                checkpoint_data = {
                    "outputs": outputs,
                    "_checkpoint_info": {
                        "processed_count": len(outputs),
                        "last_id": item_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                print(f"\n💾 Saved checkpoint ({len(outputs)} items)")

        except Exception as e:
            print(f"\n❌ Error processing ID {item_id}: {e}")

            # Save checkpoint on error
            if args.checkpoint_interval > 0:
                checkpoint_data = {
                    "outputs": outputs,
                    "_checkpoint_info": {
                        "processed_count": len(outputs),
                        "last_error": str(e),
                        "failed_id": item_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                print(f"💾 Saved error checkpoint ({len(outputs)} items)")
            raise

    # Save final output
    print(f"\n{'='*60}")
    print("Saving final results...")
    print(f"{'='*60}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved final output to: {args.output_path}")
    print(f"✅ Total items generated: {len(outputs)}")

    # Remove checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("🗑️  Checkpoint file deleted")

    print(f"\n{'='*60}")
    print("Generation completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
