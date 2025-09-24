#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import random

from datasets import (
    load_dataset,
    DatasetDict,
    concatenate_datasets,
    disable_caching,
)

disable_caching()


def is_soccer(ex):
    url = (ex.get("url") or ex.get("context_url") or "").lower()
    title = (ex.get("context_title") or "").lower()
    return ("football.fandom.com" in url) or ("football wiki" in title)  # soccer (futebol)


def is_basket(ex):
    url = (ex.get("url") or ex.get("context_url") or "").lower()
    title = (ex.get("context_title") or "").lower()
    return ("basketball.fandom.com" in url) or ("basketball wiki" in title)


def _extract_text_and_start(raw_ans, ctx: str):
    """
    Normaliza diferentes formatos de 'answers' vindos do QASports:
    - dict: {"text": "xxx" ou ["xxx"], "offset": [start, end]} OU {"text": ["xxx"], "answer_start": [start]}
    - str: "xxx"
    - list: ["xxx"] OU [{"text": "...", "offset": [...]}]
    Retorna (text, start_char).
    """
    # dict
    if isinstance(raw_ans, dict):
        text = raw_ans.get("text", "")
        if isinstance(text, list):
            text = text[0] if text else ""
        else:
            text = str(text) if text is not None else ""

        # prioridade: answer_start, depois offset, depois localização via find
        if isinstance(raw_ans.get("answer_start"), (list, tuple)) and raw_ans["answer_start"]:
            start = int(raw_ans["answer_start"][0])
            return text, max(start, 0)

        if isinstance(raw_ans.get("offset"), (list, tuple)) and len(raw_ans["offset"]) == 2:
            start = int(raw_ans["offset"][0])
            return text, max(start, 0)

        # fallback
        if text and ctx:
            pos = ctx.find(text)
            return text, (pos if pos >= 0 else 0)
        return text or "", 0

    # list
    if isinstance(raw_ans, list):
        if not raw_ans:
            return "", 0
        first = raw_ans[0]
        if isinstance(first, dict):
            return _extract_text_and_start(first, ctx)
        # assume lista de strings
        text = str(first)
        if text and ctx:
            pos = ctx.find(text)
            return text, (pos if pos >= 0 else 0)
        return text or "", 0

    # str
    if isinstance(raw_ans, str):
        text = raw_ans
        if text and ctx:
            pos = ctx.find(text)
            return text, (pos if pos >= 0 else 0)
        return text or "", 0

    # desconhecido
    return "", 0


def to_squad_like(example):
    ctx = example.get("context", "") or ""
    # Alguns dumps usam "answers", outros "answer"
    raw = example.get("answers")
    if raw is None:
        raw = example.get("answer")

    text, start = _extract_text_and_start(raw, ctx)

    return {
        "id": example.get("qa_id", "") or example.get("id", ""),
        "question": example.get("question", "") or "",
        "context": ctx,
        "answers": {"text": [text], "answer_start": [start]},
        "context_title": example.get("context_title", "") or "",
        "url": example.get("url") or example.get("context_url") or "",
    }


def sample_by_sport(ds, selector_fn, max_n=None, seed=42):
    sub = ds.filter(selector_fn)
    if (max_n is not None) and (len(sub) > max_n):
        random.seed(seed)
        idx = list(range(len(sub)))
        random.shuffle(idx)
        sub = sub.select(idx[:max_n])
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/processed/qasports_soccer_basketball")
    ap.add_argument("--max-per-sport", type=int, default=None, help="limita exemplos por esporte por split")
    ap.add_argument("--dataset", type=str, default="PedroCJardim/QASports",
                    help="alternativa menor: leomaurodesenv/QASports2")
    ap.add_argument("--splits", type=str, default="train,validation,test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.out)
    ds_dict = {}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        try:
            ds = load_dataset(args.dataset, split=split)
        except Exception:
            if split != "train":
                print(f"[warn] Split '{split}' indisponível em {args.dataset}; pulando.")
                continue
            ds = load_dataset(args.dataset, split="train")

        soccer = sample_by_sport(ds, is_soccer, max_n=args.max_per_sport, seed=args.seed)
        basket = sample_by_sport(ds, is_basket, max_n=args.max_per_sport, seed=args.seed)
        both = concatenate_datasets([soccer, basket])

        keep = {"id", "qa_id", "question", "context", "answers", "context_title", "url", "context_url"}
        remove_cols = [c for c in both.column_names if c not in keep]

        both = both.map(to_squad_like, remove_columns=remove_cols)
        ds_dict[split] = both

    if not ds_dict:
        raise SystemExit("Nenhum split carregado. Verifique o nome do dataset/splits.")

    dd = DatasetDict(ds_dict)
    out_path.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(out_path))

    for k, v in dd.items():
        print(f"[{k}] {len(v):,} exemplos")
    print(f"\nOK! Dataset salvo em: {out_path}\n")


if __name__ == "__main__":
    main()
