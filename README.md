# Audio Reasoning Model — Think-then-Answer

An end-to-end **Audio–Language Model** that ingests raw audio, produces an explicit **Chain-of-Thought reasoning trace**, and then outputs a final answer — all in a single forward pass.

## Architecture

```
audio waveform
  → frozen wav2vec2 encoder  (facebook/wav2vec2-base-960h)
  → mean-pool over time
  → linear projection  (trainable)
  → prefix conditioning  (virtual audio tokens)
  → small LLM with LoRA  (microsoft/phi-2, 2.7 B)
  → "Thinking: … \n Final Answer: …"
```

| Component | Detail |
|---|---|
| Audio encoder | `facebook/wav2vec2-base-960h` — frozen, hidden size 768 |
| Language model | `microsoft/phi-2` (2.7 B) loaded in 4-bit via bitsandbytes |
| Adaptation | LoRA (r = 16, α = 32) — only the adapter + projection are trained |
| Input sample rate | 16 000 Hz (auto-resampled) |

## Repository Contents

| File | Description |
|---|---|
| `audio_reasoning_model (1).ipynb` | Main notebook — full pipeline (data → training → inference → evaluation → JSONL output) |
| `test.py` | Standalone Python script version of the same pipeline |

## Getting Started

### Requirements

- Python 3.10+
- CUDA GPU recommended (fits a Colab T4 in 4-bit)

### Install dependencies

```bash
pip install torch torchaudio \
    "transformers>=4.40" "accelerate>=0.28" \
    "peft>=0.10" "bitsandbytes>=0.44.1" \
    librosa soundfile
```

### Run

**Notebook (Google Colab / VS Code Interactive):**

Open `audio_reasoning_model (1).ipynb` and run all cells from top to bottom. The notebook uses `# %%` cell markers and works in both Colab and VS Code.

**Script:**

```bash
python test.py
```

## Pipeline Overview

1. **Environment Setup** — installs all Python packages.
2. **Imports & Configuration** — all hyperparameters live in one `CFG` dict.
3. **Audio Frontend** — `AudioEncoder` wraps the frozen wav2vec2 model and returns a pooled embedding for any `.wav` file.
4. **Dataset** — generates a synthetic dataset of tonal audio scenes with hand-written question/reasoning/answer triples. Optionally downloads and mixes in the real-world **ESC-50** environmental sound dataset.
5. **Language Model + LoRA** — loads `microsoft/phi-2` in 4-bit and attaches a LoRA adapter.
6. **Fusion Model** — `AudioLanguageModel` projects the audio embedding into multiple virtual prefix tokens in the LLM embedding space, then auto-regressively generates the reasoning chain and answer.
7. **Training** — standard AdamW loop with gradient clipping and an optional curriculum schedule.
8. **Inference** — `generate_answer()` returns the full `Thinking: … Final Answer: …` string.
9. **Evaluation** — semantic match accuracy computed over the test split (including ESC-50 label matching).
10. **JSONL Output** — results written to `results.jsonl` with fields `id`, `thinking_prediction`, `answer_prediction`, `answer_correct`.

## Key Hyperparameters

All settings are controlled through the `CFG` dictionary at the top of the script / notebook:

| Key | Default | Description |
|---|---|---|
| `audio_encoder_name` | `facebook/wav2vec2-base-960h` | Pretrained wav2vec2 checkpoint |
| `lm_name` | `microsoft/phi-2` | Pretrained LLM checkpoint |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA scaling |
| `lr` | `1e-5` | Learning rate |
| `num_epochs` | `8` | Training epochs |
| `max_new_tokens` | `256` | Max tokens generated at inference |
| `esc50_use` | `True` | Mix in ESC-50 real-world sounds |

## Checkpoints

Trainable weights (LoRA adapter + projection layer) are saved to `./checkpoints/`:

```
checkpoints/
  lora_adapter/   ← HuggingFace PEFT format
  audio_proj.pt   ← projection layer state dict
```

Load them back with `load_checkpoint(model)`.

## Output Format

Each row in `results.jsonl`:

```json
{
  "id": "sample_0001",
  "thinking_prediction": "The first tone is low ...",
  "answer_prediction": "Pitch increases over time.",
  "answer_correct": true
}
```

## Using Your Own Audio

Replace the synthetic dataset in the *Dataset* section with real `AudioSample` entries:

```python
samples = [
    AudioSample(
        audio_path="/path/to/my_audio.wav",
        question="What sound is this?",
        reasoning="...",   # used as supervision during training
        answer="Dog bark.",
    ),
]
```

Any sample rate and channel count are accepted — the encoder resamples to 16 kHz mono automatically.

## License

See repository for license information.
