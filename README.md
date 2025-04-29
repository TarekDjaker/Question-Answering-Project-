# Projet Question Answering

Ce dépôt contient une implémentation de plusieurs approches de **Question Answering** sur le dataset [NQ-Open](https://huggingface.co/datasets/nq_open) (validation).

## Approches implémentées

1. **Retrieval-Augmented Generation (RAG)**

   - Utilise un index dense pré-calculé (`fakesearch.load("faiss-validation.pickle")`).
   - Génération via GPT-2, ChatGPT (GPT-3.5-Turbo) et Llama3 via Ollama.
   - Script principal : `rag_neural_all.py`.

2. **Closed-Book QA**

   - Appel direct aux modèles (GPT-2, Llama2, ChatGPT) sans contexte externe.
   - Script principal : `closed_book_qa.py`.

## Structure du dépôt

```
├── rag_neural_all.py       # Évaluation RAG sur 50 exemples
├── closed_book_qa.py      # Évaluation Closed-Book QA sur 50 exemples
├── fakesearch.py          # Module pour charger l'index dense précalculé
├── faiss-validation.pickle# Index dense pré-calculé pour RAG
├── requirements.txt       # Dépendances Python
└── README.md              # Ce fichier
```

## Prérequis

- Python 3.8+
- Clé API OpenAI (exportée sous `OPENAI_API_KEY`)
- Ollama installé et modèle `llama3` chargé (pour Llama3)
- Java 11+ et Pyserini (facultatif si on utilise uniquement `fakesearch`)

## Installation

```bash
git clone <url-du-dépôt>
cd Projet
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows
pip install -r requirements.txt
# Charger le modèle Llama3 si nécessaire
# ollama pull llama3
```

## Usage

### RAG QA

```bash
python rag_neural_all.py
```

- Évalue les 50 premières questions de la split `validation`.
- Affiche l'exact match de chaque générateur (GPT-2, GPT-3.5, Llama3) sur les passages récupérés.

### Closed-Book QA

```bash
python closed_book_qa.py
```

- Évalue les mêmes 50 questions en mode closed-book.
- Sauvegarde un résumé `results_summary.csv` et affiche les scores EM.

## Personnalisation

- Modifier la taille de l'échantillon : ajuster `dataset_size` ou `MAX_EXAMPLES` dans les scripts.
- Changer de modèles : éditer `OPENAI_MODEL`, `LLAMA2_MODEL` ou paramètres de génération.

## Analyse des Résultats

Les scripts affichent pour chaque approche :

- **Exact Match (%)** : proportion de réponses exactement identiques aux références.
- Nombre de questions traitées et temps d'exécution.

Une analyse comparative peut être réalisée en comparant les scores RAG vs closed-book et entre modèles.

## Licence

Ce projet est sous licence MIT.
