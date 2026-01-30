# UEval: A Benchmark for Unified Multimodal Generation

Official code of UEval: A Benchmark for Unified Multimodal Generation

> [**UEval: A Benchmark for Unified Multimodal Generation**](https://arxiv.org/abs/2502.12150) </br>
> *[Bo Li](https://primerl.github.io/), [Yida Yin](https://davidyyd.github.io), [Wenhao Chai](https://wenhaochai.com/), [Xingyu Fu](https://zeyofu.github.io/)\*, [Zhuang Liu](https://liuzhuang13.github.io)\* (* indicates co-advising) <br>
> Princeton University<br>
> [[Paper]](https://arxiv.org/abs/2601.22155) [[Project page]](https://zlab-princeton.github.io/UEval/) [[Dataset]](https://huggingface.co/datasets/zlab-princeton/UEval)

---

<p align="center">
<img src="https://github.com/user-attachments/assets/f66379a8-c571-4cba-8e9b-78d158ecd26c" width=100% height=100%
class="center">
</p>

We introduce **UEval**, a benchmark to evaluate unified models, i.e., models capable of generating both images and text. UEval comprises 1,000 expert-curated prompts that require both images and text in the model outputs, sourced from 8 diverse real-world domains.


## Generating Model Outputs

### Using Gemini API

We provide `generate_outputs/gemini.py` for generating multimodal outputs (both text and images) using Google's Gemini API.

#### Prerequisites

1. Install required packages:
```bash
pip install google-genai datasets pillow
```

2. Set up your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

#### Generate Outputs

Basic usage:
```bash
python generate_outputs/gemini.py \
  --output_path results/gemini_outputs.json \
  --output_image_dir results/images/ \
  --api_key YOUR_API_KEY
```

#### Advanced Options

```bash
# Generate for specific domains
python generate_outputs/gemini.py \
  --output_path results/gemini_outputs.json \
  --output_image_dir results/images/ \
  --domains art life tech

# Limit number of items for testing
python generate_outputs/gemini.py \
  --output_path results/test.json \
  --output_image_dir results/images/ \
  --limit 10

# Use specific Gemini model
python generate_outputs/gemini.py \
  --output_path results/gemini_outputs.json \
  --output_image_dir results/images/ \
  --model gemini-2.5-flash-image
```

**Key Arguments:**
- `--api_key`: Gemini API key (or set `GEMINI_API_KEY` environment variable)
- `--output_path`: Path to save output JSON file (required)
- `--output_image_dir`: Directory to save generated images (required)
- `--hf_dataset`: HuggingFace dataset ID (default: `primerL/UEval-all`)
- `--domains`: Filter by specific task types (e.g., `art`, `life`, `tech`, `exercise`, `space`, `textbook`, `diagram`, `paper`)
- `--model`: Gemini model name (default: `gemini-2.5-flash-image`)
- `--limit`: Number of items to process (default: all)
- `--checkpoint_interval`: Save checkpoint every N items (default: 1)
- `--retry_delay`: Seconds between retry attempts (default: 3.0)
- `--max_attempts`: Maximum retry attempts per prompt (default: 100)

#### Output Format

Generated outputs are saved in JSON format compatible with the evaluation script:
```json
[
  {
    "id": 1,
    "prompt": "Your prompt here...",
    "task_type": "art",
    "question_type": "open",
    "gemini_image_ans": ["results/images/1_1.png"],
    "gemini_text_ans": "Generated text response..."
  },
  ...
]
```

### Using Emu3.5

We adapted [Emu3.5's official implementation](https://github.com/baaivision/Emu3.5) to work with the UEval benchmark by adding two adapter files: `ueval_inference_vllm.py` and `vis_proto_ueval.py`.

#### Prerequisites

1. Follow the [official Emu3.5 setup instructions](https://github.com/baaivision/Emu3.5) to configure the environment and download model weights.

2. Ensure you have the required dependencies installed as specified in the Emu3.5 repository.

#### Generate Outputs

**Step 1: Run inference to generate protobuf outputs**

```bash
cd generate_outputs/Emu3.5
python ueval_inference_vllm.py \
  --cfg configs/example_config_visual_guidance.py \
  --dataset-name primerL/UEval-all \
```

This will generate protobuf (`.pb`) files containing the raw model outputs.

**Step 2: Visualize and convert protobuf outputs to evaluation format**

```bash
python src/utils/vis_proto_ueval.py \
  --proto-dir outputs/proto \
  --image-dir images \
  --output-json emu3.5_results.json 
```

This converts the protobuf files into JSON format compatible with the UEval evaluation script.

**Key Arguments for inference:**
- `--cfg`: Path to Emu3.5 configuration file (required)
- `--dataset-name`: HuggingFace dataset ID (default: `primerL/UEval-all`)
- `--dataset-split`: Specific split to process (e.g., `art`, `life`, etc.)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (default: 4)
- `--gpu-memory-utilization`: GPU memory utilization ratio (default: 0.7)

**Key Arguments for visualization:**
- `--proto-dir`: Directory containing `.pb` files (required)
- `--image-dir`: Directory to save extracted images (required)
- `--output-json`: Path to save output JSON file (required)
- `--relative-root`: Base directory for computing relative image paths (default: `.`)

#### Output Format

The final JSON output will have the following format:
```json
[
  {
    "id": "1",
    "emu_image": ["clip_00_00.png", ...],
    "emu_text": "Generated text response with chain-of-thought..."
  },
  ...
]
```

## Evaluation

### Quick Start

We provide `ueval_eval.py` for efficient evaluation using the Gemini API with caching strategy to reduce costs.

**Cost Estimates:**
- Without caching: ~$90 per full benchmark evaluation
- With caching: Cost savings depend on how many cached contents can be created
  - When judging reference answers: Can save ~$25 (almost every answer in open-ended questions can create cache)
  - For model outputs: Savings vary based on number of generated images per question
- **Caching requirements**: Gemini's context caching requires `min_total_token_count=2048`, typically achieved when evaluating answers with 5+ images
- Use `--no_cache` flag for models generating single images, as caching threshold may not be reached

#### Prerequisites

1. Install required packages:
```bash
pip install google-generativeai datasets pillow
```

2. Set up your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

#### Evaluate Your Model

Prepare your model outputs in JSON format:
```json
[
  {
    "id": "1",
    "text_answer": "Your model's text response",
    "image_answer": ["path/to/generated/image1.jpg", "path/to/generated/image2.jpg"]
  },
  ...
]
```

Run evaluation:
```bash
python ueval_eval.py \
  --model_output_path path/to/your_model_outputs.json \
  --output_path results/your_model_results.json \
```

#### Advanced Options

```bash
# With caching (recommended for models generating multiple images per question)
python ueval_eval.py \
  --model_output_path path/to/outputs.json \
  --output_path path/to/results.json \
  --text_field text_answer \
  --image_field image_answer \
  --api_key YOUR_API_KEY \
  --limit 10 \

# Without caching (recommended for models generating single images)
python ueval_eval.py \
  --model_output_path path/to/outputs.json \
  --output_path results/results.json \
  --text_field text_answer \
  --image_field image_answer \
  --api_key YOUR_API_KEY \
  --limit 100 \
  --no_cache
```


**Key Arguments:**
- `--model_output_path`: Path to your model's output JSON file (required)
- `--output_path`: Where to save evaluation results (required)
- `--hf_dataset`: HuggingFace dataset ID (default: `primerL/UEval-all`)
- `--text_field`: Field name for text answers in your output file (default: `text_answer`)
- `--image_field`: Field name for image paths in your output file (default: `image_answer`)
- `--api_key`: Gemini API key (or set `GEMINI_API_KEY` environment variable)
- `--limit`: Number of examples to evaluate (default: all)
- `--checkpoint_interval`: Save checkpoint every N items (default: 1)
- `--no_cache`: Disable caching for single-image outputs (optional)


### Output Format

Evaluation results are saved in JSON format:
```json
{
  "results": [
    {
      "id": "1",
      "text_rate": 0.85,
      "image_rate": 0.90,
      "text_rubrics": [...],
      "image_rubrics": [...],
      "text_evaluations": [...],
      "image_evaluations": [...]
    },
    ...
  ],
  "summary": {
    "num_items": 1000,
    "num_items_with_text": 1000,
    "num_items_with_image": 1000,
    "text_avg_rate": 0.82,
    "image_avg_rate": 0.78,
    "overall_avg_rate": 0.80
  }
}
```

## Results

We evaluate recent unified models on all 8 tasks in our benchmark. Overall, frontier models consistently outperform open-source ones across all tasks: GPT-5-Thinking achieves the highest average score of 66.4, while the best open-source model obtains only 49.1. The gap between proprietary and open-source models is very large: the strongest frontier model (e.g., GPT-5-Thinking) outperforms the best open-source model (e.g., Emu 3.5) by over 17 points on average.

| Model                           | Space    | Textbook | Diagram  | Paper    | Art      | Life     | Tech     | Exercise | Avg      |
| ------------------------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| *Reference*                     | 96.2     | 94.4     | 93.1     | 96.2     | 90.6     | 87.7     | 90.6     | 89.2     | 92.2     |
| Janus-Pro                       | 21.0     | 31.0     | 37.4     | 15.2     | 26.4     | 23.0     | 17.6     | 11.5     | 22.9     |
| Show-o2                         | 25.4     | 33.1     | 33.2     | 17.4     | 25.6     | 15.6     | 17.4     | 13.1     | 22.6     |
| MMaDA                           | 10.8     | 20.0     | 14.2     | 13.3     | 15.7     | 15.8     | 12.4     | 12.6     | 14.4     |
| BAGEL                           | 29.8     | 42.5     | 37.2     | 20.0     | 39.0     | 33.6     | 24.8     | 21.4     | 31.0     |
| Emu3.5                      | **59.1** | **57.4** | **41.1** | **31.6** | **59.3** | **62.0** | **37.0** | **45.4** | **49.1** |
| Gemini-2.0-Flash                | 65.2     | 55.2     | 47.6     | 45.8     | **70.4** | 58.0     | 50.2     | 48.0     | 55.1     |
| Gemini-2.5-Flash                | 78.0     | 74.0     | 66.4     | **71.6** | 66.6     | 63.0     | **58.2** | 50.0     | 66.0     |
| GPT-5-Instant                   | 77.3     | 77.9     | 62.3     | 55.1     | 71.2     | **69.7** | 50.7     | 57.6     | 65.2     |
| GPT-5-Thinking              | **84.0** | **78.0** | **67.8** | 51.9     | 67.8     | 63.8     | 57.0     | **61.4** | **66.4** |



<img width="1368" height="1417" alt="Image" src="https://github.com/user-attachments/assets/ede55cbf-fffd-4278-8fed-34d8c95596a8" />

<img width="1398" height="1542" alt="Image" src="https://github.com/user-attachments/assets/add410df-b267-4693-8a3d-b9c8e45108fb" />

## Citation
If you find this repository helpful, please consider citing:
```bibtex
@article{li2026ueval,
    title     = {UEval: A Benchmark for Unified Multimodal Generation},
    author    = {Li, Bo and Yin, Yida and Chai, Wenhao and Fu, Xingyu and Liu, Zhuang},
    journal   = {arXiv preprint arXiv:2601.22155},
    year      = {2026}
}
```
