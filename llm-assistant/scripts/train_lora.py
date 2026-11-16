"""Fine-tuning LoRA do Qwen/Qwen3-1.7B com máscara de perda (-100) em bfloat16."""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import BASE_MODEL, DIR_ADAPTERS, DIR_ASSETS, DIR_DATA, SYSTEM_PROMPT, load_jsonl

SEED = 42
MAX_LENGTH = 1024
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _set_seeds() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _tokenize_dataset(rows: list[dict], tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH) -> Dataset:
    def _encode(ex: dict) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["user_prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        answer = ex["target_json"] + tokenizer.eos_token
        full_text = prompt + answer

        tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        labels = tokens["input_ids"].copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        tokens["labels"] = labels
        return tokens

    return Dataset.from_list(rows).map(_encode, remove_columns=list(rows[0].keys()))


def _make_collator(pad_token_id: int):
    def _collate(features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn_masks, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_token_id] * pad_len)
            attn_masks.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return _collate


def _plot_learning_curve(history: list[dict], base_model: str, out_path) -> None:
    """Plota curva de Train vs Validation Loss em função das épocas e passos."""
    train_pts = [(h["epoch"], h["loss"]) for h in history if "loss" in h and "epoch" in h]
    eval_pts = [(h["epoch"], h["step"], h["eval_loss"]) for h in history if "eval_loss" in h and "epoch" in h]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    if train_pts:
        ax.plot([p[0] for p in train_pts], [p[1] for p in train_pts], label="Train Loss (a cada 5 passos)", marker="o", markersize=3.5)
    if eval_pts:
        for ep, _st, val in eval_pts:
            ax.axvline(ep, color="#94a3b8", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.plot([p[0] for p in eval_pts], [p[2] for p in eval_pts], label="Validation Loss (fim de época)", marker="s", markersize=6.5, linewidth=2.2)
        for i, (ep, _st, val) in enumerate(eval_pts):
            dx = -10 if i == len(eval_pts) - 1 else 6
            ha = "right" if i == len(eval_pts) - 1 else "left"
            dy = -14 if (i == 0 and val > 0.08) else 8
            va = "top" if dy < 0 else "bottom"
            ax.annotate(
                f"{val:.4f}" if val >= 0.001 else f"{val:.5f}",
                (ep, val),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va=va,
                fontsize=8.5,
                fontweight="bold",
                color="#c2410c",
            )
    model_short = base_model.split("/")[-1]
    ax.set_title(f"Curva de Aprendizado ({model_short} + LoRA r=16)")
    if eval_pts:
        xticks = [0.0] + [p[0] for p in eval_pts]
        xlabels = ["0\n(início)"] + [f"Época {int(round(ep))}\n(passo {st})" for ep, st, _ in eval_pts]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_xlim(0.0, max(xticks[-1] + 0.12, 3.12))
    ax.set_xlabel("Progresso do Treinamento (3 Épocas; lote efetivo = 16)")
    ax.set_ylabel("Cross-Entropy Loss (máscara -100)")
    max_y = max([p[1] for p in train_pts] + [p[2] for p in eval_pts] + [1.0])
    ax.set_ylim(-0.06, max_y * 1.12)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter-name", default="lora_arborgrid_r16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--history-prefix", default="")
    parser.add_argument("--replot", action="store_true")
    args_cli, _ = parser.parse_known_args()

    hist_path = DIR_ASSETS / f"{args_cli.history_prefix}lora_train_history.json"
    curve_path = DIR_ASSETS / f"{args_cli.history_prefix}curva_loss_lora.png"
    if args_cli.replot:
        history = json.loads(hist_path.read_text(encoding="utf-8"))
        _plot_learning_curve(history, args_cli.base_model, curve_path)
        print(f"Curva atualizada em {curve_path}")
        return

    _set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device} | Base: {args_cli.base_model}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM livre: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

    train_rows = load_jsonl(DIR_DATA / "sft_train.jsonl")
    val_rows = load_jsonl(DIR_DATA / "sft_val.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args_cli.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds_train = _tokenize_dataset(train_rows, tokenizer, max_length=args_cli.max_length)
    ds_val = _tokenize_dataset(val_rows, tokenizer, max_length=args_cli.max_length)
    max_target_tokens = max(sum(1 for x in row["labels"] if x != -100) for row in ds_train)
    print(f"Amostras treino={len(ds_train)}, val={len(ds_val)}, max tokens resposta={max_target_tokens}")

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        args_cli.base_model,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    model.enable_input_require_grads()
    if device == "cuda":
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    ckpt_dir = ROOT / f"checkpoints_{args_cli.adapter_name}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=3,
        per_device_train_batch_size=args_cli.batch_size,
        per_device_eval_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_accum,
        learning_rate=2e-4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        bf16=(device == "cuda"),
        report_to="none",
        seed=SEED,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=_make_collator(tokenizer.pad_token_id),
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Treinamento concluído em {elapsed:.1f}s")

    out_adapter = DIR_ADAPTERS / args_cli.adapter_name
    out_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_adapter))
    tokenizer.save_pretrained(str(out_adapter))

    history = trainer.state.log_history
    DIR_ASSETS.mkdir(parents=True, exist_ok=True)
    with hist_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2, ensure_ascii=False)

    _plot_learning_curve(history, args_cli.base_model, curve_path)

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"Adaptador salvo em {out_adapter}")


if __name__ == "__main__":
    main()
