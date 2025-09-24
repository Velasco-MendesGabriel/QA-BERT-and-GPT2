# spaces/app.py
import os
import deploy_space as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "gkovacsvelasco/distilbert-qa-qasports")
MAX_LEN = int(os.getenv("MAX_LEN", "384"))
DOC_STRIDE = int(os.getenv("DOC_STRIDE", "128"))

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(HF_MODEL_ID)
model.eval()

def answer(question, context, topk=3, max_answer_len=30):
    if not question or not context:
        return {"best_answer": "", "topk": []}

    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        truncation="only_second",
        max_length=MAX_LEN,
        stride=DOC_STRIDE,
        return_offsets_mapping=True
    )
    with torch.no_grad():
        out = model(**{k: v for k, v in inputs.items() if k in ("input_ids","attention_mask","token_type_ids")})

    start = out.start_logits[0].cpu().numpy()
    end = out.end_logits[0].cpu().numpy()

    # top-n busca combinatória simples
    n_best = 20
    start_idx = np.argsort(start)[-n_best:][::-1]
    end_idx = np.argsort(end)[-n_best:][::-1]
    offsets = inputs["offset_mapping"][0].cpu().numpy().tolist()
    # sequence_ids só funciona via tokenizer fast:
    seq_ids = tokenizer.sequence_ids()
    context_token_ids = [i for i, sid in enumerate(seq_ids) if sid == 1]

    prelim = []
    for s in start_idx:
        for e in end_idx:
            if e < s:
                continue
            if (e - s + 1) > max_answer_len:
                continue
            if s not in context_token_ids or e not in context_token_ids:
                continue
            score = float(start[s] + end[e])
            sc, _ = offsets[s]
            _, ec = offsets[e]
            if sc is None or ec is None:
                continue
            text = context[sc:ec].strip()
            if text:
                prelim.append({"text": text, "score": score})

    if not prelim:
        return {"best_answer": "", "topk": []}

    # dedup por texto + softmax em scores
    by_text = {}
    for p in prelim:
        t = p["text"]
        if t not in by_text or p["score"] > by_text[t]["score"]:
            by_text[t] = p
    nbest = sorted(by_text.values(), key=lambda x: x["score"], reverse=True)[:topk]
    scores = np.array([x["score"] for x in nbest], dtype=np.float64)
    probs = np.exp(scores - scores.max()); probs /= probs.sum()
    for i, nb in enumerate(nbest):
        nb["prob"] = float(probs[i])

    return {"best_answer": nbest[0]["text"], "topk": nbest}

with gr.Blocks(title="Sports QA (DistilBERT)") as demo:
    gr.Markdown("# 🏀⚽ Sports QA (DistilBERT)\nPergunte algo e forneça um contexto do basquete ou futebol (soccer).")
    with gr.Row():
        question = gr.Textbox(label="Pergunta")
    context = gr.Textbox(label="Contexto", lines=10)
    k = gr.Slider(1, 5, value=3, step=1, label="Top-K respostas")

    btn = gr.Button("Responder")
    best = gr.Textbox(label="Melhor resposta")
    table = gr.Dataframe(headers=["resposta", "probabilidade"], datatype=["str", "number"])

    def ui_fn(q, c, tk):
        out = answer(q, c, tk)
        rows = [[r["text"], round(r["prob"], 4)] for r in out["topk"]]
        return out["best_answer"], rows

    btn.click(ui_fn, inputs=[question, context, k], outputs=[best, table])

demo.queue()
