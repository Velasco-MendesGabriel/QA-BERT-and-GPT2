#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import collections
import json
import numpy as np
from typing import List, Tuple

from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import evaluate


# ----------------------------
# Pré-processamento (train/eval)
# ----------------------------
def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    # Tokeniza com mapeamento e janela deslizante sobre o contexto
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

        # Se a resposta não cabe no contexto desta janela → CLS
        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # Mapeia char→token
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
    # Para avaliação, precisamos manter offset_mapping e o mapeamento para exemplos originais
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
    Converte logits (start, end) em textos de resposta por exemplo,
    seguindo a lógica padrão do SQuAD (n-best + restrições).
    Retorna dict: example_id -> {"text": best_answer, "n_best": [...]}.
    """
    all_start_logits, all_end_logits = predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, f in enumerate(features):
        features_per_example[f["example_id"]].append(i)

    final_predictions = {}
    softmax = lambda x: np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

    for example_id, feature_indices in features_per_example.items():
        context = examples[examples["id"] == example_id]["context"][0]

        prelim = []
        for fi in feature_indices:
            start_logits = all_start_logits[fi]
            end_logits = all_end_logits[fi]
            offsets = features[fi]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]

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
                    _, end_char = offsets[e]
                    if start_char is None or end_char is None:
                        continue
                    text = context[start_char:end_char]
                    score = start_logits[s] + end_logits[e]
                    prelim.append({"text": text, "score": float(score)})

        if len(prelim) == 0:
            final_predictions[example_id] = {"text": "", "n_best": []}
            continue

        # Consolida duplicatas por texto (pega maior score)
        by_text = {}
        for p in prelim:
            t = p["text"].strip()
            if t == "":
                continue
            if t not in by_text or p["score"] > by_text[t]["score"]:
                by_text[t] = p

        nbest = sorted(by_text.values(), key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Probabilidades normalizadas (softmax dos scores)
        scores = np.array([x["score"] for x in nbest], dtype=np.float64)
        probs = softmax(scores)
        for i, p in enumerate(nbest):
            p["prob"] = float(probs[i])

        final_predictions[example_id] = {"text": nbest[0]["text"], "n_best": nbest}

    return final_predictions


# ----------------------------
# Utilitário: extrair Top-K por pergunta única
# ----------------------------
def topk_answers_for_single_qa(model, tokenizer, question: str, context: str, k: int = 3):
    inputs = tokenizer(question, context, return_tensors="pt", truncation="only_second", max_length=384)
    with np.no_grad:  # noqa
        outputs = model(**{k: v for k, v in inputs.items()})
    start = outputs.start_logits.detach().numpy()[0]
    end = outputs.end_logits.detach().numpy()[0]

    # Busca combinatória simples (como no postprocess) + softmax
    n_best_size = 20
    max_answer_length = 30
    start_idx = np.argsort(start)[-n_best_size:][::-1]
    end_idx = np.argsort(end)[-n_best_size:][::-1]
    prelim = []
    for s in start_idx:
        for e in end_idx:
            if e < s or (e - s + 1) > max_answer_length:
                continue
            score = start[s] + end[e]
            prelim.append((s, e, score))

    prelim.sort(key=lambda x: x[2], reverse=True)
    prelim = prelim[:n_best_size]

    # Para mapear tokens→texto, precisamos de offsets (tokenizer rápido)
    encoded = tokenizer(question, context, return_offsets_mapping=True, truncation="only_second", max_length=384)
    offsets = encoded["offset_mapping"]
    sequence_ids = encoded.sequence_ids()
    # Limita a spans somente no contexto
    context_token_ids = [i for i, sid in enumerate(sequence_ids) if sid == 1]

    candidates = []
    for s, e, score in prelim:
        if s not in context_token_ids or e not in context_token_ids:
            continue
        start_char, _ = offsets[s]
        _, end_char = offsets[e]
        text = (question + " " + context)[start_char:end_char] if start_char is not None and end_char is not None else ""
        candidates.append({"text": text, "score": float(score)})

    if not candidates:
        return []

    # Consolida + softmax
    by_text = {}
    for c in candidates:
        t = c["text"].strip()
        if t == "":
            continue
        if t not in by_text or c["score"] > by_text[t]["score"]:
            by_text[t] = c
    nbest = sorted(by_text.values(), key=lambda x: x["score"], reverse=True)[:k]
    scores = np.array([x["score"] for x in nbest], dtype=np.float64)
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
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
        remove_columns=[],
        desc="Tokenizing eval",
    )

    # Modelos e métricas
    metric = evaluate.load("squad")

    def compute_metrics(eval_pred):
        """
        Esta função é chamada após o post_process (ver abaixo),
        então recebe dicionários já com 'predictions' e 'references'.
        """
        return metric.compute(predictions=eval_pred.predictions, references=eval_pred.label_ids)

    # Pós-processamento para o Trainer (conecta features → textos)
    def post_processing_function(examples, features, predictions, stage="eval"):
        preds = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            tokenizer=tokenizer,
            n_best_size=20,
            max_answer_length=30,
        )
        # Constrói pares para a métrica SQuAD
        formatted_predictions = [{"id": k, "prediction_text": v["text"]} for k, v in preds.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return {"predictions": formatted_predictions, "label_ids": references}

    # --------------- Baseline (antes do FT) ---------------
    base_model = AutoModelForQuestionAnswering.from_pretrained(args.base_model)
    baseline_args = TrainingArguments(
        output_dir=args.output_dir + "/baseline",
        per_device_eval_batch_size=args.eval_batch,
        seed=args.seed,
        report_to="none",
    )
    baseline_trainer = Trainer(
        model=base_model,
        args=baseline_args,
        eval_dataset=eval_feats.remove_columns([c for c in eval_feats.column_names if c not in ("input_ids", "attention_mask")]),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        post_process_function=lambda examples, features, outputs: post_processing_function(
            eval_examples, eval_feats, outputs, stage="baseline"
        ),
    )
    print("\n[Baseline] Avaliando modelo base (sem fine-tuning)...")
    baseline_metrics = baseline_trainer.evaluate(eval_dataset=eval_feats)
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
        eval_dataset=eval_feats.remove_columns([c for c in eval_feats.column_names if c not in ("input_ids", "attention_mask")]),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        post_process_function=lambda examples, features, outputs: post_processing_function(
            eval_examples, eval_feats, outputs, stage="eval"
        ),
    )

    trainer.train()

    # --------------- Avaliação pós-treino ---------------
    print("\n[Fine-tuned] Avaliando modelo após fine-tuning...")
    ft_metrics = trainer.evaluate(eval_dataset=eval_feats)
    print("[Fine-tuned] Métricas:", ft_metrics)

    # --------------- Amostras com Top-3 e “confiança” ---------------
    # Seleciona N exemplos da validação e gera top-3
    n_samples = min(args.sample_predictions, len(eval_examples))
    sample_idxs = list(range(n_samples))
    samples = []
    ft_model.eval()
    for idx in sample_idxs:
        ex = eval_examples[idx]
        q = ex["question"]
        ctx = ex["context"]
        # Reaproveita o próprio modelo fine-tuned
        with np.no_grad:  # noqa
            pass
        # Implementação de topk usando logits + softmax (como no postprocess)
        inputs = tokenizer(q, ctx, return_tensors="pt", truncation="only_second", max_length=args.max_length)
        outputs = ft_model(**{k: v for k, v in inputs.items()})
        start = outputs.start_logits.detach().numpy()[0]
        end = outputs.end_logits.detach().numpy()[0]

        # Busca combinatória top-n
        n_best_size = 20
        max_answer_length = 30
        start_idx = np.argsort(start)[-n_best_size:][::-1]
        end_idx = np.argsort(end)[-n_best_size:][::-1]
        prelim = []
        for s in start_idx:
            for e in end_idx:
                if e < s or (e - s + 1) > max_answer_length:
                    continue
                prelim.append((s, e, float(start[s] + end[e])))

        prelim.sort(key=lambda x: x[2], reverse=True)
        prelim = prelim[:n_best_size]

        # offsets para recortar texto do contexto
        enc = tokenizer(q, ctx, return_offsets_mapping=True, truncation="only_second", max_length=args.max_length)
        offsets = enc["offset_mapping"]
        seq_ids = enc.sequence_ids()
        context_token_ids = [i for i, sid in enumerate(seq_ids) if sid == 1]

        cands = []
        for s, e, score in prelim:
            if s not in context_token_ids or e not in context_token_ids:
                continue
            sc, _ = offsets[s]
            _, ec = offsets[e]
            if sc is None or ec is None:
                continue
            text = (q + " " + ctx)[sc:ec] if sc < len(q + " " + ctx) and ec <= len(q + " " + ctx) else ctx[sc:ec]
            cands.append({"text": text.strip(), "score": score})

        # consolida + softmax
        dedup = {}
        for c in cands:
            t = c["text"]
            if not t:
                continue
            if t not in dedup or c["score"] > dedup[t]["score"]:
                dedup[t] = c
        nbest = sorted(dedup.values(), key=lambda x: x["score"], reverse=True)[:3]
        if nbest:
            scores = np.array([x["score"] for x in nbest], dtype=np.float64)
            probs = np.exp(scores - scores.max())
            probs /= probs.sum()
            for i, nb in enumerate(nbest):
                nb["prob"] = float(probs[i])

        samples.append({
            "id": ex["id"],
            "question": q,
            "topk": nbest,
            "gold": ex["answers"]["text"] if "answers" in ex else [],
        })

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "sample_predictions.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # --------------- Salvar/push ---------------
    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(commit_message="CI: push DistilBERT QA (subset)")

    # Loga baseline x ft para comparação
    with open(Path(args.output_dir) / "metrics_summary.json", "w") as f:
        json.dump({"baseline": baseline_metrics, "fine_tuned": ft_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
