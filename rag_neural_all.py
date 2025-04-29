
#!/usr/bin/env python3
import os, time
from typing import List, Dict, Tuple
import fakesearch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from langchain_ollama import ChatOllama
from openai import OpenAI, APIError, APIConnectionError
import torch # Import torch pour la détection GPU
import concurrent.futures # Import pour la parallélisation
from collections import Counter # Pour compter les résultats de manière thread-safe

# === Configurations ===
# Assurez-vous que votre clé API est définie comme variable d'environnement
# ou utilisez une méthode sécurisée pour la gérer.
# Par exemple: API_KEY = os.getenv("OPENAI_API_KEY")
# Pour l'exemple, je remets votre clé, mais ce n'est pas recommandé en production.
API_KEY = "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"
if API_KEY:
    client_openai = OpenAI(api_key=API_KEY)
else:
    print("Attention: Clé API OpenAI non configurée.")
    client_openai = None

# Déterminer le device (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device: {device}")

# === Global loading ===
print("Chargement de l'index dense...")
dense_index = fakesearch.load("faiss-validation.pickle")
print("Chargement du dataset...")
# Réduire la taille pour le test si nécessaire
dataset_size = 50
ds = load_dataset('nq_open', split='validation').select(range(dataset_size))

print("Chargement de GPT-2...")
tok_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
mdl_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device) # <-- Placer sur le device
if not tok_gpt2.pad_token:
    tok_gpt2.pad_token = tok_gpt2.eos_token
    mdl_gpt2.config.pad_token_id = tok_gpt2.eos_token_id

print("Initialisation de Llama3 via Ollama...")
try:
    # Assurez-vous qu'Ollama est lancé et que le modèle 'llama3' est disponible
    llm_ollama3 = ChatOllama(model='llama3', temperature=0)
    # Test rapide de connexion
    llm_ollama3.invoke("Hello")
    print("Connexion à Llama3 réussie.")
except Exception as e:
    print(f"Impossible d'initialiser ou de se connecter à Llama3 via Ollama: {e}")
    llm_ollama3 = None

# === Functions ===
def retrieve_dense(q: str, k: int = 3) -> List[str]:
    # La recherche peut être sensible à la casse, essayons les deux
    hits = dense_index.get(q) or dense_index.get(q.lower()) or []
    # Limiter la taille du contexte pour éviter de dépasser les limites des modèles
    return [item['content'].replace("\n", " ").strip()[:300] for item in hits[:k] if 'content' in item]

def gen_gpt2(q: str, ctxs: List[str]) -> str:
    prompt = 'Answer based only on context.' + ''.join(
        f" [Context {i+1}] {c}" for i, c in enumerate(ctxs)
    ) + f" Question: {q} Answer:"
    # Tronquer si le prompt est trop long pour GPT-2 (max 1024 tokens par défaut)
    inputs = tok_gpt2(prompt, return_tensors='pt', truncation=True, max_length=924).to(device) # <-- Placer les inputs sur le device
    try:
        with torch.no_grad(): # Désactiver le calcul du gradient pour l'inférence
            out = mdl_gpt2.generate(**inputs, max_new_tokens=100, pad_token_id=tok_gpt2.eos_token_id, do_sample=False)
        # S'assurer que la séquence générée est bien après le prompt
        gen_tokens = out[0, inputs['input_ids'].shape[-1]:]
        return tok_gpt2.decode(gen_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Erreur lors de la génération GPT-2: {e}")
        return ''

def gen_gpt35(q: str, ctxs: List[str]) -> str:
    if not client_openai: return '' # Ne pas continuer si le client n'est pas initialisé
    prompt = 'Answer based only on context.' + ''.join(
        f" [Context {i+1}] {c}" for i, c in enumerate(ctxs)
    ) + f" Question: {q} Answer:"
    try:
        resp = client_openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=150, temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except (APIError, APIConnectionError) as e:
        print(f"Erreur API OpenAI: {e}")
        # Optionnel: ajouter une pause et réessayer ? Pour l'instant, on retourne ''
        time.sleep(1) # Petite pause en cas d'erreur
        return ''
    except Exception as e: # Capturer d'autres erreurs potentielles
        print(f"Erreur inattendue lors de l'appel OpenAI: {e}")
        return ''


def gen_llama3(q: str, ctxs: List[str]) -> str:
    if not llm_ollama3: return '' # Ne pas continuer si Llama3 n'est pas dispo
    if not ctxs: return '' # Éviter d'appeler avec contexte vide si retrieve échoue
    prompt = 'Answer based only on context.' + ''.join(f" Context: {c}" for c in ctxs) + f" Question: {q}"
    try:
        # L'appel invoke peut être bloquant
        response = llm_ollama3.invoke(prompt)
        # La réponse peut être un objet AIMessage, extraire le contenu
        if hasattr(response, 'content'):
            return response.content.strip()
        return str(response).strip() # Fallback si ce n'est pas un AIMessage
    except Exception as e:
        print(f"Erreur lors de la génération Llama3: {e}")
        return ''

def exact_match(pred: str, truths: List[str]) -> bool:
    # Normalisation simple : minuscule et suppression des espaces de début/fin
    pred_normalized = pred.lower().strip()
    if not pred_normalized: # Une prédiction vide n'est jamais correcte
        return False
    for truth in truths:
        if pred_normalized == str(truth).lower().strip():
            return True
    return False

# === Fonction pour traiter un seul exemple ===
def process_example(ex: Dict) -> Dict[str, bool]:
    """Traite un exemple, retourne les succès pour chaque modèle."""
    q = ex['question']
    truths = list(map(str, ex['answer'])) if isinstance(ex['answer'], list) else [str(ex['answer'])]
    # Étape 1: Récupération (commune à tous les modèles)
    ctxs = retrieve_dense(q)

    results = {}
    # Étape 2 & 3: Génération et Évaluation (peut être parallélisé par modèle si besoin)
    # Pour l'instant, on les fait séquentiellement dans cette fonction,
    # mais la fonction elle-même sera appelée en parallèle pour différents exemples.
    pred_gpt2 = gen_gpt2(q, ctxs)
    results['GPT2'] = exact_match(pred_gpt2, truths)

    pred_gpt35 = gen_gpt35(q, ctxs)
    results['GPT3.5'] = exact_match(pred_gpt35, truths)

    pred_llama3 = gen_llama3(q, ctxs)
    results['Llama3'] = exact_match(pred_llama3, truths)

    # Optionnel : Afficher la progression ou les résultats intermédiaires
    # print(f"Q: {q[:50]}... | GPT2: {results['GPT2']} | GPT3.5: {results['GPT3.5']} | Llama3: {results['Llama3']}")

    return results

# === Main Evaluation avec Parallélisation ===
def main():
    total_counts = Counter() # Utilise Counter pour agréger les résultats des threads
    total_processed = 0
    start = time.time()

    # Ajuster max_workers en fonction des capacités de votre machine (CPU, RAM)
    # et des limites des API ou services (Ollama, OpenAI)
    # Un point de départ raisonnable peut être le nombre de cœurs CPU.
    num_workers = os.cpu_count() or 4 # Défaut à 4 si cpu_count() retourne None
    print(f"Démarrage de l'évaluation avec {num_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Soumettre toutes les tâches
        futures = [executor.submit(process_example, ex) for ex in ds]

        # Récupérer les résultats au fur et à mesure qu'ils sont complétés
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result() # Récupère le dict {'GPT2': bool, ...}
                total_counts.update({model: 1 for model, correct in result.items() if correct})
                total_processed += 1
                # Afficher la progression
                print(f"Progression: {total_processed}/{len(ds)} exemples traités.", end='\r')
            except Exception as e:
                print(f"\nUne tâche a échoué: {e}")

    elapsed = time.time() - start
    print(f"\n\nÉvaluation sur {total_processed} exemples terminée en {elapsed:.2f}s")
    print("== Résultats (Exact Match) ==")
    if total_processed == 0:
        print("Aucun exemple n'a été traité avec succès.")
        return

    for name in ['GPT2', 'GPT3.5', 'Llama3']:
        correct = total_counts[name]
        acc = correct / total_processed * 100
        print(f"{name}: {acc:.2f}% ({correct}/{total_processed})")

if __name__ == '__main__':
    main()


