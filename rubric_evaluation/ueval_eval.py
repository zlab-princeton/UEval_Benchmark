#!/usr/bin/env python3
"""
Data preparation:
    The evaluation dataset with rubrics is loaded automatically from HuggingFace:
    https://huggingface.co/datasets/primerL/UEval-all/

    Your model output JSON should follow this format:
    [
        {
            "id": "1",
            "text_answer": "Your model's text response...",
            "image_answer": ["path/to/image1.jpg", "path/to/image2.jpg"]
        },
        ...
    ]

    Note: The 'id' field must match the IDs in the HuggingFace dataset.
          Images can be local paths or URLs.

API Key Setup:
    Set your Gemini API key via environment variable:
    export GEMINI_API_KEY="your-api-key-here"

    Or pass it directly via command line:
    --api_key YOUR_API_KEY

Usage example:
    # Using environment variable
    export GEMINI_API_KEY="your-api-key-here"
    python eval_cache_v2.py \
        --model_output_path your_model_outputs.json \
        --output_path eval/results.json \
        --model gemini-2.5-pro \
        --cache_ttl 3600s

    # Using command line argument
    python eval_cache_v2.py \
        --api_key your-api-key-here \
        --model_output_path your_model_outputs.json \
        --output_path eval/results.json \
        --model gemini-2.5-pro \
        --no_cache

Note: Use models that support caching (e.g., gemini-2.5-pro)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from google import genai
from google.genai import types

try:
    from datasets import load_dataset
except ImportError:
    print("Warning: 'datasets' library not found. Install with: pip install datasets")
    load_dataset = None

# Gemini API configuration
# Users should provide their API key via environment variable or command line argument
SYSTEM_INSTRUCTION = """
You are an expert AI evaluator. Your job is to look at a conversation, a rubric item and assess model outputs (text and images) against specific rubric criteria.
Return a JSON object with "criteria_met". 
""".strip()

TEXT_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score how well the model's text answer satisfies the rubric.

# Conversation
Question: <<question>>
Text Answer: <<text_answer>>

# Instructions
Return a JSON object with the field:  "criteria_met" (boolean or "not sure").
- Set "criteria_met" to true only if the rubric is fully satisfied. Use false if any requirement is missing or incorrect. If there is not enough information, return "not sure".
Return only the JSON object (no extra narration).
""".strip()

TEXT_RUBRIC_TEMPLATE = """
# Rubric Item
<<rubric_item>>
""".strip()

IMAGE_TEMPLATE_OPEN = """
You are evaluating whether the generated image (considered together with the accompanying text answer) satisfies the rubric.

# Conversation
Question: <<question>>
Text Answer: <<text_answer>>

# Instructions
You are given the question, the model's text answer, and the generated image(s). Judge whether the visual content (and its alignment with the text) meets the rubric.
Return a JSON object with "criteria_met".
- Set "criteria_met" to true only if the rubric is completely satisfied; false otherwise. Use "not sure" if the evidence is insufficient.
- One important clarification regarding the requirement is that each image must include a visual depiction of the described action — it cannot rely solely on text rendered within the image as a substitute for visual content. For example, if the rubric says "Each image must directly correspond to a single, sequential step outlined in the text answer," then the image must visually represent the action described in the text (e.g., showing the motion, object, or scene), rather than merely displaying textual labels or written descriptions inside the image.
Return only the JSON object.
- One important exception to the above point is that when the criterion is used to evaluate the consistency between an image step and its corresponding text step, the image does not need to depict all actions or details mentioned in that step to meet the criterion.
For example, if the criterion states, "Each image must visually represent the primary action described in its corresponding numbered step in the text," then an image that clearly shows the main action—such as turning the oven dial to preheat—would still satisfy the criterion, even if the step also includes secondary actions (like preparing the baking tray or measuring ingredients).
The key point is that the image should accurately represent the primary action of the step, rather than all of its described details.
""".strip()

IMAGE_RUBRIC_TEMPLATE = """
# Rubric Item
<<rubric_item>>
""".strip()

IMAGE_TEMPLATE_CLOSED = """
You are evaluating whether the generated image satisfies an image-focused rubric.

# Question
<<question>>

# Instructions
You are given the question and the generated image(s). Judge whether the image meets the rubric. Return a JSON object with "criteria_met".
- Set "criteria_met" to true only if the rubric is completely satisfied; false otherwise. Use "not sure" if the evidence is insufficient.
- One important clarification regarding the image requirement is that each image must include a visual depiction of the described content — it cannot rely solely on text rendered within the image as a substitute for visual content.
Return only the JSON object. If any image consists purely of text with no visual content, it should be judged as false directly.
""".strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_from_hf(hf_dataset_id: str) -> List[Dict[str, Any]]:
    """
    Load evaluation dataset with rubrics from HuggingFace.
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
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from HuggingFace: {e}")


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v]
    return [value] if value else []


def normalize_criteria_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        lowered = cleaned.lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
        return cleaned
    return value


def criteria_value_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
        if lowered in {"not sure", "unsure", "unknown"}:
            return False
    return False


def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON from Gemini response with better error handling."""
    import re

    # Method 1: Try to find JSON code block (```json ... ```)
    json_block_match = re.search(
        r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_block_match:
        json_str = json_block_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # Fall through to other methods

    # Method 2: Find JSON object with criteria_met or explanation field
    # This avoids matching random braces in text like {nodel}
    json_pattern = re.search(
        r'\{\s*"(?:criteria_met|explanation)"[^}]*\}', response_text, re.DOTALL)
    if json_pattern:
        json_str = json_pattern.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # Fall through to original method

    # Method 3: Original method (find first { to last })
    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError(
            f"Could not locate JSON object in response:\n{response_text}")
    json_str = response_text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Print debug information
        print(f"\n❌ JSON parsing error: {e}")
        print(f"Problematic JSON string (first 500 chars):\n{json_str[:500]}")
        print(f"Full response text:\n{response_text}\n")

        # Try to fix common escape issues
        try:
            # Fix unescaped backslashes in strings (but not already escaped ones)
            fixed_json = re.sub(
                r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', json_str)
            return json.loads(fixed_json)
        except:
            # If still fails, re-raise the original error with context
            raise ValueError(
                f"Failed to parse JSON even after escape fixing. Original error: {e}\nJSON: {json_str}")


def send_to_gemini_with_parse(
    client: genai.Client,
    model_name: str,
    prompt: str,
    image_paths: Optional[Sequence[str]],
    base_dir: Path,
    max_attempts: int = 500,
    retry_delay: float = 5.0,
) -> Dict[str, Any]:
    """Send request to Gemini and parse JSON response with retry on parse failure."""
    for attempt in range(max_attempts):
        try:
            # Get response from Gemini
            response_text = send_to_gemini(
                client, model_name, prompt, image_paths, base_dir,
                max_attempts=None, retry_delay=retry_delay
            )

            # Try to parse the response
            parsed = parse_gemini_response(response_text)
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"\n⚠️  JSON parsing failed (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                print(f"Retrying in {retry_delay} seconds...\n")
                time.sleep(retry_delay)
            else:
                print(f"❌ Maximum retries reached, parsing still failed")
                raise
        except RuntimeError as e:
            print(
                f"\n⚠️  API request failed (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                print(f"Retrying in {retry_delay} seconds...\n")
                time.sleep(retry_delay)
            else:
                print(f"❌ Maximum retries reached, API still not responding")
                raise

    raise RuntimeError("Failed to get valid JSON response after all retries")


def create_text_cache(
    client: genai.Client,
    model_name: str,
    context_prompt: str,
    ttl: str = "900s",
) -> Optional[Any]:
    """
    Create a cache for text evaluation context.
    Caches: question + text_answer + TEXT_TEMPLATE
    Returns the cache object if successful, None otherwise.
    """
    try:
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                display_name=f'text_cache_{int(time.time())}',
                system_instruction=SYSTEM_INSTRUCTION,
                contents=[context_prompt],
                ttl=ttl,
            )
        )
        return cache
    except Exception as e:
        print(f"[WARN] Failed to create text cache: {e}")
        return None


def upload_images_and_create_cache(
    client: genai.Client,
    model_name: str,
    image_paths: List[str],
    base_dir: Path,
    context_prompt: str,
    ttl: str = "3600s",
) -> Optional[Any]:
    """
    Upload images to Gemini Files API and create a cache with context.
    Now caches: images + question + text_answer + template instructions
    Returns the cache object if successful, None otherwise.
    """
    if not image_paths:
        return None

    uploaded_files = []
    for rel_path in image_paths:
        img_path = Path(rel_path)
        if not img_path.is_absolute():
            img_path = base_dir / img_path
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            continue

        try:
            # Upload file
            uploaded_file = client.files.upload(file=img_path)

            # Wait for processing
            while uploaded_file.state.name == 'PROCESSING':
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)

            if uploaded_file.state.name == 'ACTIVE':
                uploaded_files.append(uploaded_file)
            else:
                print(f'[WARN] Image processing failed: {img_path.name}')

        except Exception as e:
            print(f"[WARN] Failed to upload {img_path}: {e}")

    if not uploaded_files:
        return None

    # Create cache with context prompt (question + text_answer + template)
    try:
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                display_name=f'image_cache_{int(time.time())}',
                system_instruction=SYSTEM_INSTRUCTION,
                contents=[uploaded_files, context_prompt],
                ttl=ttl,
            )
        )
        return cache
    except Exception as e:
        print(f"[WARN] Failed to create image cache: {e}")
        return None


def send_with_cache(
    client: genai.Client,
    model_name: str,
    prompt: str,
    cache: Any,
    max_attempts: int = 500,
    retry_delay: float = 5.0,
) -> Dict[str, Any]:
    """Send request using cached content."""
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(cached_content=cache.name),
            )

            response_text = getattr(response, "text", None)
            if not response_text:
                raise ValueError("Empty response")

            # Print token usage
            if hasattr(response, 'usage_metadata'):
                m = response.usage_metadata
                print(
                    f"[Cache] Prompt:{m.prompt_token_count} Cached:{getattr(m, 'cached_content_token_count', 0)} Response:{m.candidates_token_count}")

            return parse_gemini_response(response_text)

        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"\n⚠️  Parsing/request failed (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(retry_delay)
            else:
                raise

    raise RuntimeError("Failed after all retries")


def send_to_gemini(
    client: genai.Client,
    model_name: str,
    prompt: str,
    image_paths: Optional[Sequence[str]],
    base_dir: Path,
    max_attempts: Optional[int] = None,
    retry_delay: float = 5.0,
) -> str:
    contents: List[Any] = [prompt]

    if image_paths:
        for rel_path in image_paths:
            img_path = Path(rel_path)
            if not img_path.is_absolute():
                img_path = base_dir / img_path
            if not img_path.exists():
                print(f"[WARN] Image not found: {img_path}")
                continue
            image = Image.open(img_path)
            contents.append(image)

    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
            )
            response_text = getattr(response, "text", None)
            if response_text:
                return response_text
            print(
                f"[WARN] Empty response (attempt {attempt}); retrying in {retry_delay}s")
            # if attempt == 10:
            #     return {
            #         "explanation": "",
            #         "criteria_met": "error"
            #     }
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] Gemini request failed on attempt {attempt}")
            print(exc)
        if max_attempts is not None and attempt >= max_attempts:
            raise RuntimeError(
                f"Gemini API did not return a successful response after {max_attempts} retries.")
        time.sleep(retry_delay)


def get_question_type_from_task(task: str) -> str:
    """
    Determine question type based on task field.
    - art, life, tech, exercise -> "open"
    - space, textbook, diagram, paper -> "closed"
    """
    task_lower = task.lower().strip()
    open_tasks = {"art", "life", "tech", "exercise"}
    closed_tasks = {"space", "textbook", "diagram", "paper"}

    if task_lower in open_tasks:
        return "open"
    elif task_lower in closed_tasks:
        return "closed"
    else:
        # Default to "open" if task is not recognized
        return "open"


def compute_score(results: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not results:
        return {"met": 0, "total": 0, "rate": None}
    total = len(results)
    met = sum(1 for item in results if criteria_value_to_bool(
        item.get("criteria_met")))
    rate = met / total if total else None
    return {"met": met, "total": total, "rate": rate}


def evaluate_item(
    item: Dict[str, Any],
    model_text: str,
    model_images: List[str],
    client: genai.Client,
    model_name: str,
    base_dir: Path,
    use_cache: bool = True,
    cache_ttl: str = "3600s",
) -> Dict[str, Any]:
    """
    Evaluate item with per-item DUAL caching strategy:
    - Text rubrics: Create text cache (question + text_answer + TEXT_TEMPLATE)
    - Image rubrics: Create image cache (images + question + text_answer + IMAGE_TEMPLATE)
    Each item can have up to 2 caches active simultaneously, both are deleted after use.
    """
    question = item.get("prompt", "")

    # Determine question type from task field
    task = item.get("task", "")
    question_type = get_question_type_from_task(task) if task else "open"

    # Step 1: Evaluate text rubrics (with cache if enabled)
    text_results: List[Dict[str, Any]] = []
    text_cache = None
    try:
        if item.get("text_rubrics"):
            # Build context prompt (WITHOUT rubric item)
            text_context_prompt = (
                TEXT_TEMPLATE.replace("<<question>>", question)
                .replace("<<text_answer>>", model_text or "")
            )

            if use_cache and len(item.get("text_rubrics", [])) > 1:
                print(
                    f"  [Cache] Creating text cache for item {item.get('id')}")
                text_cache = create_text_cache(
                    client, model_name, text_context_prompt, ttl=cache_ttl)
                if text_cache:
                    print(f"  [Cache] Created text cache: {text_cache.name}")

            for rubric in item.get("text_rubrics", []):
                criterion = rubric.get("criterion", "")

                # Build rubric prompt
                rubric_prompt = TEXT_RUBRIC_TEMPLATE.replace(
                    "<<rubric_item>>", criterion)

                # Use cache if available, otherwise fallback to direct API
                if text_cache:
                    parsed = send_with_cache(
                        client, model_name, rubric_prompt, text_cache)
                else:
                    # Fallback: send full prompt (context + rubric)
                    full_prompt = text_context_prompt + "\n\n" + rubric_prompt
                    parsed = send_to_gemini_with_parse(
                        client, model_name, full_prompt, None, base_dir)

                print(parsed)
                criteria_met = normalize_criteria_value(
                    parsed.get("criteria_met"))

                text_results.append(
                    {
                        "criterion": rubric.get("criterion", ""),
                        "criteria_met": criteria_met,
                        "explanation": parsed.get("explanation", ""),
                        "raw_response": str(parsed),
                    }
                )
    finally:
        if text_cache:
            try:
                client.caches.delete(name=text_cache.name)
                print(f"  [Cache] Deleted text cache: {text_cache.name}")
            except Exception as exc:
                print(
                    f"[WARN] Failed to delete text cache {text_cache.name}: {exc}")

    # Step 2: Evaluate image rubrics (with per-item cache)
    image_results: List[Dict[str, Any]] = []
    if item.get("image_rubrics"):
        # Create cache ONLY for this item's images + context
        cache = None
        try:
            template = IMAGE_TEMPLATE_OPEN if question_type == "open" else IMAGE_TEMPLATE_CLOSED
            # Build context prompt (WITHOUT rubric item)
            context_prompt = (
                template.replace("<<question>>", question)
                .replace("<<text_answer>>", model_text or "")
            )

            if use_cache:
                print(
                    f"  [Cache] Uploading {len(model_images)} images + context for item {item.get('id')}")
                cache = upload_images_and_create_cache(
                    client, model_name, model_images, base_dir,
                    context_prompt=context_prompt, ttl=cache_ttl)
                if cache:
                    print(f"  [Cache] Created cache: {cache.name}")

            for rubric in item.get("image_rubrics", []):
                criterion = rubric.get("criterion", "")

                # Build rubric prompt
                rubric_prompt = IMAGE_RUBRIC_TEMPLATE.replace(
                    "<<rubric_item>>", criterion)

                # Use cache if available, otherwise fallback to direct API
                if cache:
                    parsed = send_with_cache(
                        client, model_name, rubric_prompt, cache)
                else:
                    # Fallback: send full prompt (context + rubric) with images
                    full_prompt = context_prompt + "\n\n" + rubric_prompt
                    parsed = send_to_gemini_with_parse(
                        client, model_name, full_prompt, model_images, base_dir)

                print(parsed)
                criteria_met = normalize_criteria_value(
                    parsed.get("criteria_met"))

                image_results.append(
                    {
                        "criterion": rubric.get("criterion", ""),
                        "criteria_met": criteria_met,
                        "explanation": parsed.get("explanation", ""),
                        "rubric_tags": rubric.get("tags", []),
                        "type": rubric.get("type", "image"),
                        "raw_response": str(parsed),
                    }
                )
        finally:
            if cache:
                try:
                    client.caches.delete(name=cache.name)
                    print(f"  [Cache] Deleted cache: {cache.name}")
                except Exception as exc:
                    print(f"[WARN] Failed to delete cache {cache.name}: {exc}")

    text_score = compute_score(text_results)
    image_score = compute_score(image_results)

    return {
        "id": item.get("id"),
        "question": question,
        "question_type": question_type,
        "text_answer": model_text,
        "image_outputs": model_images,
        "text_results": text_results,
        "image_results": image_results,
        "text_score": text_score,
        "image_score": image_score,
    }


def compute_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    text_rates = [
        itm["text_score"]["rate"]
        for itm in items
        if itm["text_score"]["rate"] is not None
    ]
    image_rates = [
        itm["image_score"]["rate"]
        for itm in items
        if itm["image_score"]["rate"] is not None
    ]

    summary = {
        "num_items": len(items),
        "text_avg_rate": sum(text_rates) / len(text_rates) if text_rates else None,
        "image_avg_rate": sum(image_rates) / len(image_rates) if image_rates else None,
        "num_items_with_text": len(text_rates),
        "num_items_with_image": len(image_rates),
    }
    return summary


def build_model_lookup(
    outputs: List[Dict[str, Any]],
    text_field: str,
    image_field: str,
) -> Dict[Any, Tuple[str, List[str]]]:
    lookup: Dict[Any, Tuple[str, List[str]]] = {}
    for entry in outputs:
        item_id = entry.get("id")
        if item_id is None:
            continue
        # Convert ID to string for consistent matching with HuggingFace dataset
        item_id_str = str(item_id)
        text_value = entry.get(text_field, "")
        image_value = entry.get(image_field)
        image_list = ensure_list(image_value)
        lookup[item_id_str] = (text_value, image_list)
    return lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate text/image rubrics for a dataset using Gemini. "
                    "Dataset with rubrics is loaded from HuggingFace automatically."
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="Gemini API key (if not provided, reads from GEMINI_API_KEY environment variable).",
    )
    parser.add_argument("--model_output_path", required=True,
                        help="Path to your model outputs JSON with format: "
                             "[{id, text_answer, image_answer}, ...]")
    parser.add_argument("--output_path", required=True,
                        help="Where to save evaluation results.")
    parser.add_argument(
        "--hf_dataset",
        default="primerL/UEval-all",
        help="HuggingFace dataset ID (default: primerL/UEval-all).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Gemini model name (default: gemini-2.5-pro).",
    )
    parser.add_argument(
        "--text_field",
        default="text_answer",
        help="Field name in model outputs that contains the generated text (default: text_answer).",
    )
    parser.add_argument(
        "--image_field",
        default="image_answer",
        help="Field name in model outputs that contains generated image path(s) (default: image_answer).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally evaluate only the first N items.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="Save checkpoint every N items (default: 1, set to 0 to disable).",
    )
    parser.add_argument(
        "--cache_ttl",
        default="900s",
        help="Cache TTL (e.g., '300s' for 5 min, '3600s' for 1 hour).",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable per-item caching (use direct API calls).",
    )
    return parser.parse_args()


def main() -> None:
    start_time = time.time()  # Start timing
    args = parse_args()
    base_dir = Path(os.getcwd())

    # Get API key from argument or environment variable
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key is required. Provide it via --api_key argument "
            "or set GEMINI_API_KEY environment variable."
        )

    # Load dataset from HuggingFace
    dataset = load_dataset_from_hf(args.hf_dataset)
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list of items.")

    # Load model outputs
    model_outputs = load_json(Path(args.model_output_path))

    output_lookup = build_model_lookup(
        model_outputs, args.text_field, args.image_field)

    client = genai.Client(api_key=api_key)
    model_name = args.model

    items_to_process = dataset[: args.limit] if args.limit else dataset

    use_cache = not args.no_cache
    if use_cache:
        print("\n✅ DUAL per-item caching enabled")
        print(f"   - Text rubrics: Per-item text cache (if 2+ rubrics)")
        print(f"   - Image rubrics: Per-item image cache")
        print(f"   - Cache TTL: {args.cache_ttl}")
        print(f"   - Each item can have up to 2 caches active simultaneously\n")
    else:
        print("\nℹ️  Caching disabled: All requests use direct API\n")

    # Checkpoint file management
    output_path = Path(args.output_path)
    checkpoint_path = output_path.parent / \
        f".{output_path.stem}_checkpoint.json"

    # Try to load existing checkpoint
    results: List[Dict[str, Any]] = []
    processed_ids = set()
    start_index = 0

    if checkpoint_path.exists() and args.checkpoint_interval > 0:
        print(f"📂 Found checkpoint file: {checkpoint_path}")
        try:
            checkpoint_data = load_json(checkpoint_path)
            results = checkpoint_data.get("items", [])
            processed_ids = {item["id"] for item in results}
            start_index = len(results)
            print(
                f"✅ Loaded {len(results)} evaluated items, continuing from item {start_index + 1}\n")
        except Exception as e:
            print(
                f"⚠️ Failed to load checkpoint: {e}, starting from beginning")
            results = []
            processed_ids = set()
            start_index = 0

    # Evaluate items
    for idx, item in enumerate(items_to_process):
        item_id = item.get("id")

        # Skip if already processed
        if item_id in processed_ids:
            print(f"⏭️  Skipping already evaluated item ID {item_id}")
            continue

        text_answer, image_paths = output_lookup.get(item_id, ("", []))
        if not text_answer and item.get("question_type") == "open":
            print(
                f"[WARN] Missing text answer for id={item_id}; proceeding with empty string.")

        print(f"\n[{idx + 1}/{len(items_to_process)}] Evaluating ID {item_id}")

        try:
            evaluation = evaluate_item(
                item=item,
                model_text=text_answer,
                model_images=image_paths,
                client=client,
                model_name=model_name,
                base_dir=base_dir,
                use_cache=use_cache,
                cache_ttl=args.cache_ttl,
            )
            results.append(evaluation)

            # Save checkpoint periodically
            if args.checkpoint_interval > 0 and (len(results) % args.checkpoint_interval == 0):
                checkpoint_data = {
                    "items": results,
                    "summary": compute_summary(results),
                    "_checkpoint_info": {
                        "processed_count": len(results),
                        "last_id": item_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                print(f"💾 Saved checkpoint ({len(results)} items)")

        except Exception as e:
            print(f"❌ Error evaluating ID {item_id}: {e}")
            # Save checkpoint on error
            if args.checkpoint_interval > 0:
                checkpoint_data = {
                    "items": results,
                    "summary": compute_summary(results) if results else {},
                    "_checkpoint_info": {
                        "processed_count": len(results),
                        "last_error": str(e),
                        "failed_id": item_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                print(f"💾 Saved error checkpoint ({len(results)} items)")
            raise

    summary = compute_summary(results)

    output_payload = {
        "items": results,
        "summary": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Remove checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"\n🗑️  Checkpoint file deleted")

    # Calculate runtime
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    print("\n==== Evaluation Summary ====")
    print(f"Items evaluated: {summary['num_items']}")
    if summary["text_avg_rate"] is not None:
        print(
            f"Average text score: {summary['text_avg_rate']:.4f} "
            f"(over {summary['num_items_with_text']} items)"
        )
    else:
        print("No text rubrics evaluated.")
    if summary["image_avg_rate"] is not None:
        print(
            f"Average image score: {summary['image_avg_rate']:.4f} "
            f"(over {summary['num_items_with_image']} items)"
        )
    else:
        print("No image rubrics evaluated.")

    print(f"\n⏱️  Total runtime: {hours:02d}:{minutes:02d}:{seconds:.2f}")


if __name__ == "__main__":
    main()
