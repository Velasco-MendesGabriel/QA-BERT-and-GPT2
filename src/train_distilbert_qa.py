#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import collections
import json
import numpy as np
from typing import List, Tuple
import re
# --- Métricas locais: EM e F1 no estilo SQuAD ---
_punc_re = re.compile(r"[^\w\s]", flags=re.UNICODE)

def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    # remove pontuação
    s = _punc_re.sub(" ", s)
    # remove artigos simples (inglês)
    s = " ".join(w for w in s.split() if w not in {"a", "an", "the"})
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
        # pega o melhor contra múltiplas referências (se houver)
        em_i = max(_exact_match_score(pred_text, g) for g in gold_texts) if gold_texts else 0.0
        f1_i = max(_f1_score(pred_text, g) for g in gold_texts) if gold_texts else 0.0
        em += float(em_i)
        f1 += float(f1_i)
        n += 1
    if n == 0:
        return {"exact_match": 0.0, "f1": 0.0}
    return {"exact_match": 100.0 * em / n, "f1": 100.0 * f1 / n}

import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)



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
    # tokenização com offsets (necessário para recortar do contexto)
    enc = tokenizer(
        question, context,
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
        stride=128,
        return_offsets_mapping=True
    )
    with torch.no_grad():  # << TROCA NP.NO_GRAD POR TORCH.NO_GRAD
        out = model(**{kk: vv for kk, vv in enc.items() if kk in ("input_ids","attention_mask","token_type_ids")})

    start = out.start_logits[0].cpu().numpy()
    end   = out.end_logits[0].cpu().numpy()

    # candidatos
    n_best_size = 20
    max_answer_length = 30
    start_idx = np.argsort(start)[-n_best_size:][::-1]
    end_idx   = np.argsort(end)[-n_best_size:][::-1]

    # offsets e ids de sequência (para filtrar só tokens do contexto)
    offsets = enc["offset_mapping"][0].cpu().numpy().tolist()
    seq_ids = enc.sequence_ids(0)  # requer tokenizer rápido
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
            text = context[sc:ec].strip()  # << RECORTE SEMPRE DO CONTEXTO
            if not text:
                continue
            score = float(start[s] + end[e])
            prelim.append({"text": text, "score": score})

    if not prelim:
        return []

    # dedup + softmax
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
        remove_columns=eval_examples.column_names,  # remove as colunas originais (evita ArrowInvalid)
        desc="Tokenizing eval",
    )

    # Modelos e métricas
    def compute_metrics(eval_pred):
        """
        Recebe o resultado do post_process_function:
          - eval_pred.predictions: [{"id", "prediction_text"}]
          - eval_pred.label_ids : [{"id", "answers": {...}}]
        """
        return squad_em_f1(eval_pred.predictions, eval_pred.label_ids)

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

        # Tokenização com offsets para recorte do CONTEXTO
        enc = tokenizer(
            q, ctx,
            return_tensors="pt",
            truncation="only_second",
            max_length=args.max_length,
            stride=args.doc_stride,
            return_offsets_mapping=True
        )
        with torch.no_grad():  # << TROCA NP.NO_GRAD POR TORCH.NO_GRAD
            out = ft_model(**{kk: vv for kk, vv in enc.items() if kk in ("input_ids","attention_mask","token_type_ids")})

        start = out.start_logits[0].cpu().numpy()
        end   = out.end_logits[0].cpu().numpy()

        # Busca combinatória top-n
        n_best_size = 20
        max_answer_length = 30
        start_idx = np.argsort(start)[-n_best_size:][::-1]
        end_idx   = np.argsort(end)[-n_best_size:][::-1]

        # offsets + sequence_ids(0) para filtrar só tokens do CONTEXTO
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
                text = ctx[sc:ec].strip()  # << RECORTE DO CONTEXTO
                if not text:
                    continue
                score = float(start[s] + end[e])
                prelim.append({"text": text, "score": score})

        # dedup + softmax
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
