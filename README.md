# Securing LLMs: Medical Chatbot Defense Pipeline

This project presents a secure, cost-efficient defense pipeline for large language model (LLM) applications, demonstrated through a Retrieval-Augmented Generation (RAG)-based medical chatbot. The framework systematically mitigates prompt injection attacks—malicious inputs designed to manipulate LLM outputs or extract sensitive data.

By combining preventive and detective strategies into a layered defense pipeline, the system maximizes attacker cost while minimizing defender cost, ensuring both security and usability in high-stakes domains such as healthcare.

## Setup / Environment variables

1. Set your LLM API key (if using OpenAI):
export GENAI_API_KEY="your_key_here"
(or set it in Windows PowerShell / cmd appropriately)


## How to run

1) Create the memory/vector store (RAG index)
- Purpose: Converts your knowledge base into embeddings and stores them in a vector index that medi_bot.py will use for retrieval.
- Run:
  python create_memory_for_LLM.py

2) Run the LLM-based defense pipeline
- Purpose: Executes automated prompts (from your datasets) through the defensive pipeline and logs the results.
- Run:
  python medi_bot.py

## Data files (quick summary)

- data/my_Dataset.csv — adversarial prompt list (used to test attack-resilience).
- data/medicine_related.csv — benign medical prompts (used to test utility of the pipeline).
- data/testing.csv — used to analyze and choose the cosine similarity threshold during development.


