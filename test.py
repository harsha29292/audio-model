# %% [markdown]
# # Audio Reasoning Model — Think-then-Answer
#
# **Goal:** A single end-to-end Audio–Language Model that takes raw audio, produces
# an explicit Chain-of-Thought reasoning trace, and then a final answer — all in one
# forward pass.
#
# **Architecture:**
# ```
# audio waveform
#   → frozen wav2vec2 encoder
#   → mean-pool over time
#   → linear projection (trainable)
#   → prefix conditioning (virtual audio token)
#   → small text LLM with LoRA
#   → "Thinking: … Final Answer: …"
# ```
#
# **Run this file** in Google Colab (GPU runtime) or VS Code Interactive (`# %%` cells).

# %% [markdown]
# ## 1 · Environment Setup

# %%
# ── Install dependencies (run once) ──────────────────────────────────────────
import subprocess, sys

_PACKAGES = [
    "torch", "torchaudio",
    "transformers>=4.40", "accelerate>=0.28",
    "peft>=0.10", "bitsandbytes>=0.43",
    "librosa", "soundfile",
]

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q"] + _PACKAGES,
    stdout=subprocess.DEVNULL,
)
print("✓ packages installed")

# %% [markdown]
# ## 2 · Imports & Configuration

# %%
import os
import json
import math
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf

from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ── Hyperparameters (single place to edit) ───────────────────────────────────
CFG = {
    # audio
    "audio_encoder_name": "facebook/wav2vec2-base-960h",
    "target_sr": 16_000,
    # language model
    "lm_name": "microsoft/phi-2",          # 2.7 B — fits Colab T4 in 4-bit
    "max_seq_len": 512,
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # training
    "lr": 2e-5,
    "weight_decay": 0.01,
    "num_epochs": 8,
    "grad_clip": 1.0,
    # inference
    "temperature": 0.2,
    "top_p": 0.9,
    "max_new_tokens": 256,
    # evaluation
    "eval_runs": 3,
    # paths
    "data_dir": "./synthetic_data",
    "output_path": "./results.jsonl",
    "ckpt_dir": "./checkpoints",
}

# %% [markdown]
# ## 3 · Audio Frontend (frozen wav2vec2)

# %%
class AudioEncoder(nn.Module):
    """
    Frozen wav2vec2 encoder.
    Input  : path to a .wav file (any sample-rate, any channels)
    Output : single audio vector of shape (1, hidden_size=768)
    """

    def __init__(self, model_name: str = CFG["audio_encoder_name"]):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze every parameter
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.hidden_size: int = self.encoder.config.hidden_size  # 768

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def load_audio(path: str, target_sr: int = CFG["target_sr"]) -> torch.Tensor:
        """Load audio file → mono float32 tensor at target_sr."""
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        return waveform.squeeze(0)  # (num_samples,)

    # ── forward ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(self, audio_path: str) -> torch.Tensor:
        waveform = self.load_audio(audio_path)
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=CFG["target_sr"],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        hidden = self.encoder(**inputs).last_hidden_state          # (1, T, 768)
        pooled = hidden.mean(dim=1)                                # (1, 768)
        return pooled


# Quick smoke test
_ae = AudioEncoder()
print(f"✓ AudioEncoder  |  hidden_size = {_ae.hidden_size}")
del _ae

# %% [markdown]
# ## 4 · Dataset Format & Synthetic Data

# %%
@dataclass
class AudioSample:
    """One audio-reasoning example."""
    audio_path: str
    question: str
    reasoning: str
    answer: str


def create_synthetic_dataset(out_dir: str = CFG["data_dir"]) -> List[AudioSample]:
    """Generate simple sine-tone .wav files with hand-written QA pairs."""
    os.makedirs(out_dir, exist_ok=True)
    sr = CFG["target_sr"]

    specs = [
        dict(
            freq=440, dur=2.0, name="tone_440hz",
            q="What type of sound is this?",
            r=(
                "The audio contains a steady, pure sinusoidal tone with no harmonics "
                "or background noise. The pitch is consistent throughout and corresponds "
                "to approximately 440 Hz, which is the standard concert pitch A4."
            ),
            a="A pure sine tone at 440 Hz (concert A).",
        ),
        dict(
            freq=261, dur=1.5, name="tone_261hz",
            q="Is this sound higher or lower in pitch than concert A (440 Hz)?",
            r=(
                "The audio contains a pure tone at a noticeably lower frequency than "
                "440 Hz. The pitch corresponds to roughly 261 Hz, which is middle C (C4). "
                "Since 261 Hz < 440 Hz, the sound is lower in pitch."
            ),
            a="Lower in pitch.",
        ),
        dict(
            freq=880, dur=1.0, name="tone_880hz",
            q="Describe the pitch of this audio relative to middle C.",
            r=(
                "The audio is a pure tone at a high frequency, significantly above "
                "middle C (261 Hz). It is approximately 880 Hz — one octave above "
                "concert A4 and well above middle C."
            ),
            a="Much higher than middle C.",
        ),
        dict(
            freq=200, dur=3.0, name="tone_200hz",
            q="Is this a short or long audio clip?",
            r=(
                "The audio plays a continuous low-frequency tone for about 3 seconds. "
                "Compared to typical short clips under 1 second, this is relatively long."
            ),
            a="Long (approximately 3 seconds).",
        ),
        dict(
            freq=1000, dur=0.5, name="tone_1000hz",
            q="What frequency range does this sound fall into?",
            r=(
                "The audio is a short, high-pitched pure tone in the upper-mid "
                "frequency range — approximately 1000 Hz."
            ),
            a="Mid-to-high frequency range (approximately 1000 Hz).",
        ),
    ]

    samples: List[AudioSample] = []
    for s in specs:
        t = np.linspace(0, s["dur"], int(sr * s["dur"]), dtype=np.float32)
        wav = (0.5 * np.sin(2 * np.pi * s["freq"] * t)).astype(np.float32)
        path = os.path.join(out_dir, f"{s['name']}.wav")
        sf.write(path, wav, sr)
        samples.append(AudioSample(path, s["q"], s["r"], s["a"]))

    # Persist metadata for reproducibility
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump([asdict(s) for s in samples], f, indent=2)

    print(f"✓ Created {len(samples)} synthetic samples in {out_dir}/")
    return samples


dataset = create_synthetic_dataset()
for s in dataset:
    print(f"  {Path(s.audio_path).name:20s}  Q: {s.question[:60]}")

# %% [markdown]
# ## 5 · Language Model + LoRA

# %%
def _find_linear_names(model: nn.Module) -> List[str]:
    """Return unique short names of all nn.Linear layers (excluding lm_head)."""
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, (nn.Linear, torch.nn.Linear)):
            short = n.split(".")[-1]
            if short != "lm_head":
                names.add(short)
    return sorted(names)


def load_language_model(
    model_name: str = CFG["lm_name"],
) -> Tuple[nn.Module, AutoTokenizer]:
    """Load a 4-bit quantized causal LM and wrap it with LoRA."""

    # ── 4-bit quantization config ────────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # ── LoRA ─────────────────────────────────────────────────────────────
    target_modules = _find_linear_names(model)
    print(f"  LoRA target modules: {target_modules}")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


lm_model, tokenizer = load_language_model()
LM_DIM: int = lm_model.config.hidden_size
print(f"✓ LM loaded  |  hidden_size = {LM_DIM}")

# %% [markdown]
# ## 6 · Audio–Text Fusion Model

# %%
class AudioLanguageModel(nn.Module):
    """
    Fuses a frozen audio encoder with a LoRA-adapted LLM via a single
    trainable linear projection that maps the audio vector into a
    'virtual prefix token' in the LLM's embedding space.
    """

    def __init__(
        self,
        audio_encoder: AudioEncoder,
        lm_model: nn.Module,
        lm_tokenizer: AutoTokenizer,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.lm = lm_model
        self.tok = lm_tokenizer

        audio_dim = audio_encoder.hidden_size             # 768
        lm_dim = lm_model.config.hidden_size              # e.g. 2560

        # Trainable projection: audio space → LLM embedding space
        self.audio_proj = nn.Linear(audio_dim, lm_dim, dtype=torch.float32)

        # Cache the device of the LLM embedding layer for alignment
        self._embed_device = next(
            lm_model.get_input_embeddings().parameters()
        ).device

    # ── prompt helpers ───────────────────────────────────────────────────
    @staticmethod
    def _make_prefix(question: str) -> str:
        return f"Question: {question}\n\nThinking:\n"

    @staticmethod
    def _make_target(reasoning: str, answer: str) -> str:
        return f"{reasoning}\n\nFinal Answer:\n{answer}"

    # ── build inputs & labels ────────────────────────────────────────────
    def _prepare_training_inputs(
        self, audio_path: str, question: str, reasoning: str, answer: str,
    ):
        """Return (combined_embeds, combined_attn, labels) with proper masking."""
        # 1. Audio prefix token
        audio_vec = self.audio_encoder(audio_path)                     # (1, 768)
        audio_tok = self.audio_proj(audio_vec.float())                 # (1, lm_dim)
        audio_tok = audio_tok.unsqueeze(1)                             # (1, 1, lm_dim)

        # 2. Tokenize question prefix and target separately
        prefix_str = self._make_prefix(question)
        target_str = self._make_target(reasoning, answer)

        prefix_ids = self.tok(
            prefix_str, return_tensors="pt", add_special_tokens=True,
        )["input_ids"]                                                 # (1, P)

        target_ids = self.tok(
            target_str, return_tensors="pt", add_special_tokens=False,
        )["input_ids"]                                                 # (1, C)

        # Append EOS so the model learns when to stop
        eos = torch.tensor([[self.tok.eos_token_id]], dtype=target_ids.dtype)
        target_ids = torch.cat([target_ids, eos], dim=1)               # (1, C+1)

        input_ids = torch.cat([prefix_ids, target_ids], dim=1)         # (1, P+C+1)

        # Truncate if needed
        max_text = CFG["max_seq_len"] - 1  # reserve 1 for audio token
        input_ids = input_ids[:, :max_text]

        prefix_len = prefix_ids.shape[1]

        # 3. Embed text
        input_ids = input_ids.to(self._embed_device)
        text_embeds = self.lm.get_input_embeddings()(input_ids)        # (1, S, lm_dim)

        # 4. Concatenate [audio_token, text_tokens]
        audio_tok = audio_tok.to(dtype=text_embeds.dtype, device=text_embeds.device)
        combined = torch.cat([audio_tok, text_embeds], dim=1)          # (1, 1+S, lm_dim)

        # 5. Attention mask (all ones)
        attn = torch.ones(combined.shape[:2], dtype=torch.long,
                          device=combined.device)

        # 6. Labels — mask audio token + question prefix
        labels = input_ids.clone()
        labels[:, :prefix_len] = -100                                  # mask question
        pad_label = torch.full(
            (1, 1), -100, dtype=labels.dtype, device=labels.device,
        )
        labels = torch.cat([pad_label, labels], dim=1)                 # (1, 1+S)

        return combined, attn, labels

    # ── build inference inputs ───────────────────────────────────────────
    def _prepare_inference_inputs(self, audio_path: str, question: str):
        """Return (combined_embeds, combined_attn) for generation."""
        audio_vec = self.audio_encoder(audio_path)
        audio_tok = self.audio_proj(audio_vec.float()).unsqueeze(1)

        prefix_str = self._make_prefix(question)
        tokens = self.tok(prefix_str, return_tensors="pt", add_special_tokens=True)
        input_ids = tokens["input_ids"].to(self._embed_device)
        text_embeds = self.lm.get_input_embeddings()(input_ids)

        audio_tok = audio_tok.to(dtype=text_embeds.dtype, device=text_embeds.device)
        combined = torch.cat([audio_tok, text_embeds], dim=1)
        attn = torch.ones(combined.shape[:2], dtype=torch.long,
                          device=combined.device)
        return combined, attn

    # ── forward (training) ───────────────────────────────────────────────
    def forward(
        self,
        audio_path: str,
        question: str,
        reasoning: Optional[str] = None,
        answer: Optional[str] = None,
    ):
        if reasoning is not None and answer is not None:
            embeds, attn, labels = self._prepare_training_inputs(
                audio_path, question, reasoning, answer,
            )
            return self.lm(inputs_embeds=embeds, attention_mask=attn, labels=labels)
        else:
            return self._prepare_inference_inputs(audio_path, question)


# Instantiate
model = AudioLanguageModel(AudioEncoder(), lm_model, tokenizer)
print(f"✓ AudioLanguageModel ready  |  projection: {model.audio_proj}")

# %% [markdown]
# ## 7 · Training

# %%
def train_model(
    model: AudioLanguageModel,
    dataset: List[AudioSample],
    num_epochs: int = CFG["num_epochs"],
    lr: float = CFG["lr"],
) -> AudioLanguageModel:
    """
    Supervised fine-tuning.  Trains ONLY:
      • audio projection layer
      • LoRA adapter weights
    Everything else (wav2vec2, base LM) stays frozen.
    """

    # Collect trainable params
    trainable = (
        list(model.audio_proj.parameters())
        + [p for _, p in model.lm.named_parameters() if p.requires_grad]
    )
    n_train = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {n_train:,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=CFG["weight_decay"])

    model.train()
    best_loss = float("inf")
    patience, patience_limit = 0, 3

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0

        # Shuffle dataset each epoch
        order = np.random.permutation(len(dataset))

        for idx in order:
            s = dataset[idx]
            optimizer.zero_grad()

            out = model(
                audio_path=s.audio_path,
                question=s.question,
                reasoning=s.reasoning,
                answer=s.answer,
            )
            loss = out.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=CFG["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(dataset)
        tag = ""

        # Simple early stopping
        if avg < best_loss - 1e-4:
            best_loss = avg
            patience = 0
            tag = " ★"
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"  Early stop at epoch {epoch} (patience={patience_limit})")
                break

        print(f"  Epoch {epoch:>2}/{num_epochs}  loss={avg:.4f}{tag}")

    model.eval()
    return model


model = train_model(model, dataset)

# %% [markdown]
# ## 8 · Inference & Output Parsing

# %%
@torch.no_grad()
def generate_answer(
    model: AudioLanguageModel,
    audio_path: str,
    question: str,
    max_new_tokens: int = CFG["max_new_tokens"],
    temperature: float = CFG["temperature"],
    top_p: float = CFG["top_p"],
) -> str:
    """
    Single-pass generation.
    Returns the full 'Thinking: … \\n Final Answer: …' string.
    """
    model.eval()
    embeds, attn = model(audio_path, question)

    gen_kwargs = dict(
        inputs_embeds=embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        pad_token_id=model.tok.pad_token_id,
        eos_token_id=model.tok.eos_token_id,
    )

    output_ids = model.lm.generate(**gen_kwargs)
    continuation = model.tok.decode(output_ids[0], skip_special_tokens=True)

    # The prompt already ends with "Thinking:\n", so the continuation is the
    # reasoning + final answer.  Reconstruct the full output for parsing.
    full = f"Thinking:\n{continuation}"
    return full


def parse_output(text: str) -> Dict[str, str]:
    """
    Parse the structured output into thinking / answer fields.
    Robust to minor formatting variations.
    """
    thinking, answer = "", ""

    if "Thinking:" in text:
        remainder = text.split("Thinking:", 1)[1]
        if "Final Answer:" in remainder:
            t_part, a_part = remainder.split("Final Answer:", 1)
            thinking = t_part.strip()
            answer = a_part.strip()
        else:
            thinking = remainder.strip()
    elif "Final Answer:" in text:
        answer = text.split("Final Answer:", 1)[1].strip()

    return {"thinking": thinking, "answer": answer}


# Quick inference test on the first sample
_test = dataset[0]
_out = generate_answer(model, _test.audio_path, _test.question)
print("── Sample inference ──")
print(_out)
print()
_parsed = parse_output(_out)
print(f"  Parsed thinking : {_parsed['thinking'][:80]}…")
print(f"  Parsed answer   : {_parsed['answer'][:80]}")

# %% [markdown]
# ## 9 · Evaluation

# %%
def evaluate(
    model: AudioLanguageModel,
    dataset: List[AudioSample],
    num_runs: int = CFG["eval_runs"],
) -> List[Dict]:
    """
    Run inference multiple times per sample.
    Assess answer correctness, stability, and reasoning quality.
    """
    results: List[Dict] = []

    for i, sample in enumerate(dataset):
        runs = []
        for _ in range(num_runs):
            raw = generate_answer(model, sample.audio_path, sample.question)
            runs.append(parse_output(raw))

        answers   = [r["answer"] for r in runs]
        thinkings = [r["thinking"] for r in runs]

        # Stability checks
        answer_stable  = len(set(answers)) == 1
        thinking_stable = len(set(thinkings)) <= 2  # allow minor variation

        # Correctness (soft substring match)
        majority_answer = max(set(answers), key=answers.count)
        gt = sample.answer.lower().strip().rstrip(".")
        pred = majority_answer.lower().strip().rstrip(".")
        answer_correct = (gt in pred) or (pred in gt)

        # Categorize
        if answer_correct and thinking_stable:
            cat = "correct_reasoning_correct_answer"
        elif answer_correct and not thinking_stable:
            cat = "weak_reasoning_correct_answer"
        elif not answer_correct and thinking_stable:
            cat = "correct_reasoning_wrong_answer"
        else:
            cat = "weak_reasoning_wrong_answer"

        result = {
            "idx": i,
            "audio_path": sample.audio_path,
            "question": sample.question,
            "ground_truth": sample.answer,
            "predicted_answer": majority_answer,
            "predicted_thinking": runs[0]["thinking"],
            "answer_correct": answer_correct,
            "answer_stable": answer_stable,
            "thinking_stable": thinking_stable,
            "category": cat,
            "all_runs": runs,
        }
        results.append(result)

        status = "✓" if answer_correct else "✗"
        print(f"  [{status}] Q{i}: {sample.question[:50]}…  →  {cat}")

    # Summary
    n = len(results)
    n_correct = sum(r["answer_correct"] for r in results)
    n_stable  = sum(r["answer_stable"] for r in results)
    print(f"\n  Accuracy  : {n_correct}/{n} ({100*n_correct/n:.0f}%)")
    print(f"  Stability : {n_stable}/{n} ({100*n_stable/n:.0f}%)")

    return results


print("── Evaluation ──")
eval_results = evaluate(model, dataset)

# %% [markdown]
# ## 10 · JSONL Output

# %%
def save_results_jsonl(
    results: List[Dict],
    path: str = CFG["output_path"],
) -> None:
    """Write results in JSONL format (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            entry = {
                "id": f"sample_{r['idx']:04d}",
                "thinking_prediction": r["predicted_thinking"],
                "answer_prediction": r["predicted_answer"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✓ Saved {len(results)} results → {path}")


save_results_jsonl(eval_results)

# Show contents
print("\n── results.jsonl ──")
with open(CFG["output_path"]) as f:
    for line in f:
        print(line.rstrip())

# %% [markdown]
# ## Optional · Save / Load Checkpoint

# %%
def save_checkpoint(model: AudioLanguageModel, path: str = CFG["ckpt_dir"]):
    """Save the trainable components (LoRA adapter + projection layer)."""
    os.makedirs(path, exist_ok=True)
    # LoRA adapter
    model.lm.save_pretrained(os.path.join(path, "lora_adapter"))
    # Projection layer
    torch.save(
        model.audio_proj.state_dict(),
        os.path.join(path, "audio_proj.pt"),
    )
    print(f"✓ Checkpoint saved → {path}/")


def load_checkpoint(model: AudioLanguageModel, path: str = CFG["ckpt_dir"]):
    """Reload trainable components from disk."""
    from peft import PeftModel
    proj_path = os.path.join(path, "audio_proj.pt")
    model.audio_proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
    print(f"✓ Checkpoint loaded ← {path}/")
    return model


save_checkpoint(model)

# %% [markdown]
# ---
# **Done.**  The full pipeline — data → training → inference → evaluation → JSONL
# output — is complete.  To use your own audio data, replace the synthetic dataset
# in Section 4 with real `AudioSample` entries pointing to `.wav` files.
