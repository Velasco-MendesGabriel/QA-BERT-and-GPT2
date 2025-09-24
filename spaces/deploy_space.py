import os, deploy_space as gr, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "gkovacsvelasco/distilbert-qa-qasports")
MAX_LEN     = int(os.getenv("MAX_LEN", "384"))
DOC_STRIDE  = int(os.getenv("DOC_STRIDE", "128"))

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=True)
model     = AutoModelForQuestionAnswering.from_pretrained(HF_MODEL_ID)
model.eval()

def answer(question, context, topk=3, max_answer_len=30):
    if not question or not context:
        return {"best": "", "topk": []}
    enc = tokenizer(
        question, context,
        return_tensors="pt",
        truncation="only_second",
        max_length=MAX_LEN,
        stride=DOC_STRIDE,
        return_offsets_mapping=True
    )
    with torch.no_grad():
        out = model(**{k:v for k,v in enc.items() if k in ("input_ids","attention_mask","token_type_ids")})
    start = out.start_logits[0].cpu().numpy()
    end   = out.end_logits[0].cpu().numpy()

    n_best = 20
    start_idx = np.argsort(start)[-n_best:][::-1]
    end_idx   = np.argsort(end)[-n_best:][::-1]
    offsets   = enc["offset_mapping"][0].cpu().numpy().tolist()

    seq_ids = tokenizer(
        question, context,
        truncation="only_second",
        max_length=MAX_LEN,
        stride=DOC_STRIDE,
        return_offsets_mapping=False
    ).sequence_ids()
    ctx_tokens = [i for i, sid in enumerate(seq_ids) if sid == 1]

    prelim = []
    for s in start_idx:
        for e in end_idx:
            if e < s or (e - s + 1) > max_answer_len:
                continue
            if s not in ctx_tokens or e not in ctx_tokens:
                continue
            sc, _ = offsets[s]
            _, ec = offsets[e]
            if sc is None or ec is None:
                continue
            text  = context[sc:ec].strip()
            if not text:
                continue
            score = float(start[s] + end[e])
            prelim.append({"text": text, "score": score})

    if not prelim:
        return {"best": "", "topk": []}

    by_text = {}
    for p in prelim:
        t = p["text"]
        if t not in by_text or p["score"] > by_text[t]["score"]:
            by_text[t] = p
    nbest = sorted(by_text.values(), key=lambda x: x["score"], reverse=True)[:topk]
    arr = np.array([x["score"] for x in nbest], dtype=np.float64)
    prob = np.exp(arr - arr.max()); prob /= prob.sum()
    for i, nb in enumerate(nbest):
        nb["prob"] = float(prob[i])

    return {"best": nbest[0]["text"], "topk": nbest}

with gr.Blocks(title="Sports QA (DistilBERT)") as demo:
    gr.Markdown("# 🏀⚽ Sports QA (DistilBERT)")
    q = gr.Textbox(label="Pergunta")
    ctx = gr.Textbox(label="Contexto", lines=10)
    k = gr.Slider(1, 5, value=3, step=1, label="Top-K respostas")
    btn = gr.Button("Responder")
    best = gr.Textbox(label="Melhor resposta")
    table = gr.Dataframe(headers=["resposta","probabilidade"], datatype=["str","number"])

    def ui(q_, c_, k_):
        out = answer(q_, c_, k_)
        rows = [[r["text"], round(r["prob"], 4)] for r in out["topk"]]
        return out["best"], rows

    btn.click(ui, inputs=[q, ctx, k], outputs=[best, table])

demo.queue()
