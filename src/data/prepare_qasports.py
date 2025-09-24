#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepara subconjuntos de QASports (soccer + basketball) para QA extrativo.

- Fonte: PedroCJardim/QASports (contém soccer/basketball/american football).
- Filtro: inclui apenas Football Wiki (soccer) e Basketball Wiki.
- Saída: DatasetDict salvo em disco (Arrow) e pronto para usar no Trainer do HF.
- Amostragem: opcional (--max-per-sport) para acelerar experimentos locais/CI.

Citações:
- QASports: https://huggingface.co/datasets/PedroCJardim/QASports
"""

import argparse
from datasets import load_dataset, DatasetDict, concatenate_datasets
from datasets import disable_caching
from pathlib import Path
import random

disable_caching()  # evita cache global se você quiser forçar rebuild

SOCCER_KEYS = ("football.fandom.com", "Football Wiki")
BASKET_KEYS = ("basketball.fandom.com", "Basketball Wiki")

def is_soccer(ex):
    url = (ex.get("url") or ex.get("context_url") or "").lower()
    title = (ex.get("context_title") or "").lower()
    return ("football.fandom.com" in url) or ("football wiki" in title)  # soccer

def is_basket(ex):
    url = (ex.get("url") or ex.get("context_url") or "").lower()
    title = (ex.get("context_title") or "").lower()
    return ("basketball.fandom.com" in url) or ("basketball wiki" in title)

def to_squad_like(example):
    """
    Converte para o formato que o Trainer de QA espera:
      - fields: question (str), context (str), answers: {"text": [str], "answer_start": [int]}
    QASports usa 'answers' com 'text' e 'offset' [start, end] OU 'answer' singular.
    """
    # Campo de resposta pode ser 'answers' (dict) OU 'answer' (dict)
    ans = example.get("answers") or example.get("answer") or {}
    text = ans.get("text", "")
    # Offset pode ser [start, end] ou ausente
    if "offset" in ans and isinstance(ans["offset"], (list, tuple)) and len(ans["offset"]) == 2:
        start = int(ans["offset"][0])
    else:
        # fallback simples (se não houver offset, tenta localizar substring)
        # Para segurança, usa .find(); se não achar, define 0
        ctx = example.get("context", "")
        start = ctx.find(text) if text else 0
        if start < 0: start = 0

    return {
        "id": example.get("qa_id", ""),
        "question": example.get("question", ""),
        "context": example.get("context", ""),
        "answers": {"text": [text], "answer_start": [start]},
        "context_title": example.get("context_title", ""),
        "url": example.get("url") or example.get("context_url") or "",
    }

def sample_by_sport(ds, selector_fn, max_n=None, seed=42):
    sub = ds.filter(selector_fn)
    if (max_n is not None) and (len(sub) > max_n):
        random.seed(seed)
        idx = list(range(len(sub)))
        random.shuffle(idx)
        idx = idx[:max_n]
        sub = sub.select(idx)
    return sub

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/processed/qasports_soccer_basketball")
    ap.add_argument("--max-per-sport", type=int, default=None,
                    help="Limita exemplos por esporte por split (ex.: 20000)")
    ap.add_argument("--dataset", type=str, default="PedroCJardim/QASports",
                    help="Alternativa: leomaurodesenv/QASports2 (menor)")
    ap.add_argument("--splits", type=str, default="train,validation,test",
                    help="Splits a carregar (comma-separated). Tente 'train,validation,test' ou só 'train'")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.out)

    # Carrega e filtra por split
    ds_dict = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        try:
            ds = load_dataset(args.dataset, split=split)
        except Exception:
            # fallback: alguns datasets só têm 'train'
            if split != "train":
                print(f"[warn] Split '{split}' indisponível em {args.dataset}; pulando.")
                continue
            ds = load_dataset(args.dataset, split="train")

        soccer = sample_by_sport(ds, is_soccer, max_n=args.max_per_sport, seed=args.seed)
+       basket = sample_by_sport(ds, is_basket, max_n=args.max_per_sport, seed=args.seed)
+       both = concatenate_datasets([soccer, basket])

        # Converte para squad-like
        both = both.map(to_squad_like, remove_columns=[c for c in both.column_names if c not in ("id", "question", "context", "answers", "context_title", "url")])

        ds_dict[split] = both

    if not ds_dict:
        raise SystemExit("Nenhum split carregado. Verifique o nome do dataset/splits.")

    dd = DatasetDict(ds_dict)
    out_path.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(out_path))

    # Resumo
    for k, v in dd.items():
        print(f"[{k}] {len(v):,} exemplos")

    print(f"\nOK! Dataset salvo em: {out_path}\n")

if __name__ == "__main__":
    main()