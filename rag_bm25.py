# rag_bm25.py
# Installation nécessaire : pip install pyserini openai faiss-cpu
# Pyserini nécessite aussi Java 11+ installé.
import os
import json
from typing import List

import openai
from pyserini.search.lucene import LuceneSearcher

# --- Configuration ---
PYSERINI_INDEX_NAME = 'msmarco-v1-passage'
OPENAI_MODEL = 'gpt-3.5-turbo'
MAX_CONTEXT_DOCS = 5
MAX_RESPONSE_TOKENS = 150

# --- Initialisation OpenAI ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or os.getenv('openai_api_key')
if not OPENAI_API_KEY:
    raise ValueError('Clé OpenAI non définie. Exportez OPENAI_API_KEY dans votre environnement.')
openai.api_key = OPENAI_API_KEY

# --- Initialisation Pyserini ---
try:
    searcher = LuceneSearcher.from_prebuilt_index(PYSERINI_INDEX_NAME)
except Exception as e:
    raise RuntimeError(f"Impossible de charger l'index Pyserini '{PYSERINI_INDEX_NAME}': {e}")

# --- Récupération de documents BM25 ---
def retrieve_docs(query: str, k: int = MAX_CONTEXT_DOCS) -> List[str]:
    hits = searcher.search(query, k=k)
    docs: List[str] = []
    for hit in hits:
        doc = searcher.doc(hit.docid)
        raw = doc.raw() if hasattr(doc, 'raw') else str(doc)
        try:
            data = json.loads(raw)
            content = data.get('contents') or data.get('raw') or raw
        except json.JSONDecodeError:
            content = raw
        docs.append(content)
    return docs

# --- Génération de réponse avec OpenAI ---
def generate_answer(question: str, contexts: List[str]) -> str:
    prompt = 'Answer the question based *only* on the context below.'
    for i, ctx in enumerate(contexts, 1):
        snippet = ctx[:500].replace('\n', ' ')
        prompt += f"\n\n[Context {i}] {snippet}"
    prompt += f"\n\nQuestion: {question}\nAnswer:"

    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=MAX_RESPONSE_TOKENS,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# --- Main ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RAG BM25 avec Pyserini + OpenAI')
    parser.add_argument('--question', '-q', type=str, help='Question à poser', required=False)
    args = parser.parse_args()

    question = args.question or input('Question: ')
    contexts = retrieve_docs(question)
    if not contexts:
        print('Aucun contexte trouvé pour la question.')
        exit(1)

    answer = generate_answer(question, contexts)
    print('\n--- Réponse ---')
    print(answer)
    print('\n--- Contextes utilisés ---')
    for i, ctx in enumerate(contexts, 1):
        snippet = ctx[:300].replace('\n', ' ')
        print(f'[Context {i}] {snippet}...')
