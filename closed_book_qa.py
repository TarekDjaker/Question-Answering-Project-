#!/usr/bin/env python3
import os, time, traceback, re, csv, string
import torch
from typing import List, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI, APIError, APIConnectionError
from langchain_ollama import ChatOllama
from tqdm import tqdm

# --- Configuration ---
MAX_EXAMPLES = 50
DATASET_NAME = "nq_open"
DATASET_SPLIT = "validation"
OPENAI_MODEL = "gpt-3.5-turbo"
LLAMA2_MODEL = "meta-llama/Llama-2-7b-chat-hf"
GPT2_MODEL = "gpt2"
LLAMA3_MODEL = "llama3"
HF_MAX_NEW_TOKENS = 50
OPENAI_MAX_TOKENS = 50

PROMPT_TEMPLATE = (
    "Based only on your internal knowledge, provide a concise answer "
    "to the following question. Output only the answer itself.\n\n"
    "Question: {question}\nAnswer:"
)

# --- Normalization & Exact Match ---
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = ''.join(c for c in s if c not in string.punctuation)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def exact_match(pred: str, truths: List[str]) -> bool:
    pn = normalize_text(pred)
    return any(normalize_text(t) == pn for t in truths)

# --- Models Initialization ---
openai_client = OpenAI(api_key="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")

tok2, mdl2 = None, None
try:
    tok2 = AutoTokenizer.from_pretrained(GPT2_MODEL)
    mdl2 = AutoModelForCausalLM.from_pretrained(GPT2_MODEL)
    if not tok2.pad_token:
        tok2.pad_token = tok2.eos_token
        mdl2.config.pad_token_id = tok2.eos_token_id
    mdl2.eval()
except Exception:
    tok2 = mdl2 = None

tokl, mdll = None, None
try:
    tokl = AutoTokenizer.from_pretrained(LLAMA2_MODEL)
    mdll = AutoModelForCausalLM.from_pretrained(
        LLAMA2_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    if not tokl.pad_token:
        tokl.pad_token = tokl.eos_token
        mdll.config.pad_token_id = tokl.eos_token_id
    mdll.eval()
except Exception:
    tokl = mdll = None

llm3 = None
try:
    llm3 = ChatOllama(model=LLAMA3_MODEL, temperature=0)
except Exception:
    llm3 = None

# --- Answering Functions ---
def answer_openai(question: str) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question)
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except (APIError, APIConnectionError):
        return ""


def answer_gpt2(question: str) -> str:
    if not tok2 or not mdl2:
        return ""
    prompt = PROMPT_TEMPLATE.format(question=question)
    inputs = tok2(prompt, return_tensors="pt", truncation=True).to(mdl2.device)
    with torch.no_grad():
        out = mdl2.generate(**inputs, max_new_tokens=HF_MAX_NEW_TOKENS,
                             pad_token_id=tok2.eos_token_id)
    txt = tok2.decode(out[0], skip_special_tokens=True)
    return txt[len(prompt):].strip()


def answer_llama2(question: str) -> str:
    if not tokl or not mdll:
        return ""
    prompt = PROMPT_TEMPLATE.format(question=question)
    inputs = tokl(prompt, return_tensors="pt", truncation=True).to(mdll.device)
    with torch.no_grad():
        out = mdll.generate(**inputs, max_new_tokens=HF_MAX_NEW_TOKENS,
                             pad_token_id=tokl.eos_token_id)
    txt = tokl.decode(out[0], skip_special_tokens=True)
    return txt[len(prompt):].strip()


def answer_llama3(question: str) -> str:
    if not llm3:
        return ""
    prompt = PROMPT_TEMPLATE.format(question=question)
    try:
        return llm3.invoke(prompt).strip()
    except Exception:
        return ""

# --- Evaluation ---
def evaluate(name: str, fn) -> Tuple[str,int,int,float,List[Tuple[str,str,List[str]]]]:
    subset = load_dataset(DATASET_NAME, split=DATASET_SPLIT).select(range(MAX_EXAMPLES))
    correct = total = 0
    mismatches: List[Tuple[str,str,List[str]]] = []
    for ex in tqdm(subset, desc=name):
        q = ex['question']
        truths = ex['answer'] if isinstance(ex['answer'], list) else [ex['answer']]
        pred = fn(q)
        if not pred:
            continue
        total += 1
        if exact_match(pred, truths):
            correct += 1
        else:
            mismatches.append((q, pred, truths))
    acc = (correct/total*100) if total else 0.0
    return name, correct, total, acc, mismatches

if __name__ == '__main__':
    results = []
    for name, fn in [
        ("GPT-3.5", answer_openai),
        ("GPT-2", answer_gpt2),
        ("LLaMA 2", answer_llama2),
        ("LLaMA 3", answer_llama3)
    ]:
        name, c, t, acc, mism = evaluate(name, fn)
        results.append((name,c,t,acc))
        print(f"\n{name}: {c}/{t} EM ({acc:.2f}%)")
        if mism:
            print(f"-- Mismatches for {name}: ")
            for q,p,gt in mism[:5]:
                print(f"Q: {q}\nP: {p}\nGT: {gt}\n{'-'*40}")
    # Save CSV
    with open('results_summary.csv','w',newline='',encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Model","Correct","Total","Accuracy (%)"]);
        w.writerows([(n,c,t,a) for n,c,t,a in results])
    print("\nDone.")
