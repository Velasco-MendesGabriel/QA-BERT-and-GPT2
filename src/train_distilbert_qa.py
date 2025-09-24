#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import collections
import json
import numpy as np
from typing import Tuple
import re
import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

# --- Métricas locais: EM e F1 no estilo SQuAD ---
_punc_re = re.compile(r"[^\w\s]", flags=re.UNICODE)

def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = _punc_re.sub(" ", s)  # remove pontuação
    s = " ".join(w for w in s.split() if w not in {"a", "an", "the"})  # remove artigos simples (inglês)
    return " ".join(s.split())

def _f1_score(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if len(p_toks) == 0 and len(g_toks) == 0:
        return 1.0
    if len(p_toks) == 0 or len(g_toks) == 0:
        return 0.0
    common = {}
    for t in p_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)

def _exact_match_score(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)

def squad_em_f1(predictions, references):
    """
    predictions: [{"id": "...", "prediction_text": "..."}]
    references : [{"id": "...", "answers": {"text": [...], "answer_start": [...]}}]
    """
    ref_map = {r["id"]: r["answers"]["text"] for r in references}
    em, f1, n = 0.0, 0.0, 0
    for p in predictions:
        pid = p["id"]
        pred_text = p.get("prediction_text", "")
        gold_texts = ref_map.get(pid, [""])
        em_i = max(_exact_match_score(pred_text, g) for g in gold_texts) if gold_texts else 0.0
        f1_i = max(_f1_score(pred_text, g) for g in gold_texts) if gold_texts else 0.0
        em += float(em_i)
        f1 += float(f1_i)
        n += 1
    if n == 0:
        return {"exact_match": 0.0, "f1": 0.0}
    return {"exact_match": 100.0 * em / n, "f1": 100.0 * f1 / n}

# ----------------------------
# Pré-processamento (train/eval)
# ----------------------------
def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]

        # Localiza os índices do contexto no input tokenizado
        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1

        if len(answers["text"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            idx_start = context_start
            while idx_start < len(offsets) and offsets[idx_start][0] <= start_char:
                idx_start += 1
            start_positions.append(idx_start - 1)

            idx_end = context_end
            while idx_end >= 0 and offsets[idx_end][1] >= end_char:
                idx_end -= 1
            end_positions.append(idx_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []

    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_idx])

    # Mantém offset_mapping apenas para tokens do contexto; zera o resto
    offset_mapping = tokenized["offset_mapping"]
    sequence_ids_list = [tokenized.sequence_ids(i) for i in range(len(tokenized["input_ids"]))]

    new_offsets = []
    for offsets, sequence_ids in zip(offset_mapping, sequence_ids_list):
        new_offsets.append([
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(offsets)
        ])
    tokenized["offset_mapping"] = new_offsets
    return tokenized

# ----------------------------
# Pós-processamento (SQuAD)
# ----------------------------
def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    tokenizer,
    n_best_size=20,
    max_answer_length=30,
):
    """
    Converte logits (start, end) em textos de resposta por exemplo.
    Retorna dict: example_id -> {"text": best_answer, "n_best": [...]}.
    """
    all_start_logits, all_end_logits = predictions

    # Mapeia example_id -> índice no Dataset 'examples'
    example_id_to_index = {ex_id: i for i, ex_id in enumerate(examples["id"])}

    # Para cada feature, guarda quais índices pertencem a cada example_id
    features_per_example = collections.defaultdict(list)
    for i, f in enumerate(features):
        # 'features[i]' funciona; mas iterar já entrega dicts linha-a-linha
        ex_id = f["example_id"]
        features_per_example[ex_id].append(i)

    final_predictions = {}
    softmax = lambda x: np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

    for example_id, feature_indices in features_per_example.items():
        # Acessa o contexto pelo índice inteiro
        ex_idx = example_id_to_index[example_id]
        context = examples[ex_idx]["context"]

        prelim = []
        for fi in feature_indices:
            start_logits = all_start_logits[fi]
            end_logits   = all_end_logits[fi]
            offsets      = features[fi]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
            end_indexes   = np.argsort(end_logits)[-n_best_size:][::-1]

            for s in start_indexes:
                for e in end_indexes:
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s:
                        continue
                    length = e - s + 1
                    if length > max_answer_length:
                        continue
                    start_char, _ = offsets[s]
                    _, end_char   = offsets[e]
                    if start_char is None or end_char is None:
                        continue
                    text  = context[start_char:end_char]
                    score = float(start_logits[s] + end_logits[e])
                    if text.strip():
                        prelim.append({"text": text, "score": score})

        if not prelim:
            final_predictions[example_id] = {"text": "", "n_best": []}
            continue

        # Dedup por texto + softmax
        by_text = {}
        for p in prelim:
            t = p["text"].strip()
            if not t:
                continue
            if t not in by_text or p["score"] > by_text[t]["score"]:
                by_text[t] = p

        nbest = sorted(by_text.values(), key=lambda x: x["score"], reverse=True)[:n_best_size]
        scores = np.array([x["score"] for x in nbest], dtype=np.float64)
        probs  = softmax(scores)
        for i, p in enumerate(nbest):
            p["prob"] = float(probs[i])

        final_predictions[example_id] = {"text": nbest[0]["text"], "n_best": nbest}
    return final_predictions

# ----------------------------
# Utilitário: extrair Top-K por pergunta única
# ----------------------------
def topk_answers_for_single_qa(model, tokenizer, question: str, context: str, k: int = 3):
    enc = tokenizer(
        question, context,
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
        stride=128,
        return_offsets_mapping=True
    )
    with torch.no_grad():
        out = model(**{kk: vv for kk, vv in enc.items() if kk in ("input_ids","attention_mask","token_type_ids")})

    start = out.start_logits[0].cpu().numpy()
    end   = out.end_logits[0].cpu().numpy()

    n_best_size = 20
    max_answer_length = 30
    start_idx = np.argsort(start)[-n_best_size:][::-1]
    end_idx   = np.argsort(end)[-n_best_size:][::-1]

    offsets = enc["offset_mapping"][0].cpu().numpy().tolist()
    seq_ids = enc.sequence_ids(0)  # tokenizer rápido
    ctx_token_ids = {i for i, sid in enumerate(seq_ids) if sid == 1}

    prelim = []
    for s in start_idx:
        for e in end_idx:
            if e < s or (e - s + 1) > max_answer_length:
                continue
            if s not in ctx_token_ids or e not in ctx_token_ids:
                continue
            sc, _ = offsets[s]
            _, ec = offsets[e]
            if sc is None or ec is None:
                continue
            text = context[sc:ec].strip()
            if not text:
                continue
            score = float(start[s] + end[e])
            prelim.append({"text": text, "score": score})

    if not prelim:
        return []

    by_text = {}
    for p in prelim:
        t = p["text"]
        if t not in by_text or p["score"] > by_text[t]["score"]:
            by_text[t] = p
    nbest = sorted(by_text.values(), key=lambda x: x["score"], reverse=True)[:k]
    scores = np.array([x["score"] for x in nbest], dtype=np.float64)
    probs = np.exp(scores - scores.max()); probs /= probs.sum()
    for i, nb in enumerate(nbest):
        nb["prob"] = float(probs[i])
    return nbest

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Diretório salvo pelo prepare_qasports.py")
    ap.add_argument("--base_model", type=str, default="distilbert-base-uncased")
    ap.add_argument("--output_dir", type=str, default="outputs/distilbert_qa")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--train_batch", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--doc_stride", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hub_model_id", type=str, default=None, help="ex.: gkovacsvelasco/distilbert-qa-qasports")
    ap.add_argument("--hub_private", action="store_true")
    ap.add_argument("--eval_split", type=str, default="validation")
    ap.add_argument("--sample_predictions", type=int, default=5, help="quantos exemplos salvar com top-3 respostas")
    args = ap.parse_args()

    data: DatasetDict = load_from_disk(args.data_dir)
    if args.eval_split not in data:
        data[args.eval_split] = data["train"].select(range(min(1000, len(data["train"]))))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # Tokeniza para treino
    train_feats = data["train"].map(
        lambda ex: prepare_train_features(ex, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=data["train"].column_names,
        desc="Tokenizing train",
    )

    # Tokeniza para avaliação (mantém mapeamentos)
    eval_examples = data[args.eval_split]
    eval_feats = eval_examples.map(
        lambda ex: prepare_validation_features(ex, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=eval_examples.column_names,
        desc="Tokenizing eval",
    )

    # ---------- Helper de avaliação (predict → postprocess → EM/F1) ----------
    def evaluate_model(model, eval_examples, eval_feats, prefix="eval"):
        eval_ds_for_trainer = eval_feats.remove_columns(
            [c for c in eval_feats.column_names if c not in ("input_ids", "attention_mask")]
        )
        tmp_trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=os.path.join(args.output_dir, f"{prefix}_tmp"),
                per_device_eval_batch_size=args.eval_batch,
                report_to="none",
                seed=args.seed,
            ),
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
        preds = tmp_trainer.predict(eval_ds_for_trainer)
        pp = postprocess_qa_predictions(
            examples=eval_examples,
            features=eval_feats,
            predictions=preds.predictions,  # (start_logits, end_logits)
            tokenizer=tokenizer,
            n_best_size=20,
            max_answer_length=30,
        )
        formatted_predictions = [{"id": k, "prediction_text": v["text"]} for k, v in pp.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
        return squad_em_f1(formatted_predictions, references)

    # --------------- Baseline (antes do FT) ---------------
    base_model = AutoModelForQuestionAnswering.from_pretrained(args.base_model)
    print("\n[Baseline] Avaliando modelo base (sem fine-tuning)...")
    baseline_metrics = evaluate_model(base_model, eval_examples, eval_feats, prefix="baseline")
    print("[Baseline] Métricas:", baseline_metrics)

    # --------------- Fine-Tuning ---------------
    ft_model = AutoModelForQuestionAnswering.from_pretrained(args.base_model)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private,
        report_to="none",
    )
    trainer = Trainer(
        model=ft_model,
        args=training_args,
        train_dataset=train_feats,
        eval_dataset=eval_feats.remove_columns(
            [c for c in eval_feats.column_names if c not in ("input_ids", "attention_mask")]
        ),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=None,  # métricas faremos no evaluate_model() abaixo
    )

    trainer.train()

    # --------------- Avaliação pós-treino ---------------
    print("\n[Fine-tuned] Avaliando modelo após fine-tuning...")
    ft_metrics = evaluate_model(ft_model, eval_examples, eval_feats, prefix="finetuned")
    print("[Fine-tuned] Métricas:", ft_metrics)

    # --------------- Amostras com Top-3 e “confiança” ---------------
    n_samples = min(args.sample_predictions, len(eval_examples))
    samples = []
    ft_model.eval()
    for idx in range(n_samples):
        ex = eval_examples[idx]
        q, ctx = ex["question"], ex["context"]
        enc = tokenizer(
            q, ctx,
            return_tensors="pt",
            truncation="only_second",
            max_length=args.max_length,
            stride=args.doc_stride,
            return_offsets_mapping=True
        )
        with torch.no_grad():
            out = ft_model(**{kk: vv for kk, vv in enc.items() if kk in ("input_ids","attention_mask","token_type_ids")})

        start = out.start_logits[0].cpu().numpy()
        end   = out.end_logits[0].cpu().numpy()

        n_best_size = 20
        max_answer_length = 30
        start_idx = np.argsort(start)[-n_best_size:][::-1]
        end_idx   = np.argsort(end)[-n_best_size:][::-1]

        offsets = enc["offset_mapping"][0].cpu().numpy().tolist()
        seq_ids = enc.sequence_ids(0)
        context_token_ids = {i for i, sid in enumerate(seq_ids) if sid == 1}

        prelim = []
        for s in start_idx:
            for e in end_idx:
                if e < s or (e - s + 1) > max_answer_length:
                    continue
                if s not in context_token_ids or e not in context_token_ids:
                    continue
                sc, _ = offsets[s]
                _, ec = offsets[e]
                if sc is None or ec is None:
                    continue
                text = ctx[sc:ec].strip()
                if not text:
                    continue
                score = float(start[s] + end[e])
                prelim.append({"text": text, "score": score})

        dedup = {}
        for c in prelim:
            t = c["text"]
            if not t:
                continue
            if t not in dedup or c["score"] > dedup[t]["score"]:
                dedup[t] = c
        nbest = sorted(dedup.values(), key=lambda x: x["score"], reverse=True)[:3]
        if nbest:
            scores = np.array([x["score"] for x in nbest], dtype=np.float64)
            probs = np.exp(scores - scores.max()); probs /= probs.sum()
            for i, nb in enumerate(nbest):
                nb["prob"] = float(probs[i])

        samples.append({
            "id": ex["id"],
            "question": q,
            "topk": nbest,
            "gold": ex.get("answers", {}).get("text", []),
        })

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "sample_predictions.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # --------------- Salvar/push ---------------
    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(commit_message="CI: push DistilBERT QA (subset)")

    with open(Path(args.output_dir) / "metrics_summary.json", "w") as f:
        json.dump({"baseline": baseline_metrics, "fine_tuned": ft_metrics}, f, indent=2)

if __name__ == "__main__":
    main()
