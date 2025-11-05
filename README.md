# Medical-Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) style medical chatbot and an LLM-based defense pipeline created by the author. The system is designed to (1) build a vector store (memory) from a knowledge base, and (2) run automated prompts (benign and adversarial) through an LLM-based defense pipeline to evaluate and mitigate adversarial prompts.

IMPORTANT: This README describes the minimal steps to run the code and how the dataset files are used. Adjust paths and environment variables as needed for your setup.

## Repository overview

- create_memory_for_LLM.py — Build a vector store from the knowledge base (creates the RAG memory/vector index).
- medi_bot.py — Run the automated prompt tests through the LLM-based defense pipeline implemented by the author.
- data/
  - my_Dataset.csv — List of adversarial prompt examples created by the author (used to evaluate attacks).
  - medicine_related.csv — List of benign medical prompts (used to test utility and correctness of the pipeline).
  - testing.csv — Used during development to find the cosine similarity threshold used in the defense decisions; its outputs informed the threshold currently set in the code.
- (Other scripts and supporting files may be present in the repo.)

## Requirements

- Python 3.8+
- Typical packages used in RAG/LLM workflows (install the exact versions from your repository's requirements file if available). Example dependencies commonly used:
  - pandas
  - numpy
  - sentence-transformers (or other embedding model)
  - faiss-cpu (or other vector store implementation)
  - langchain (optional / if used)
  - openai (or other LLM client) — set up API key as required

If a requirements.txt is not present, create a virtual environment and install packages as needed:
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install pandas numpy sentence-transformers faiss-cpu openai

## Setup / Environment variables

1. Set your LLM API key (if using OpenAI):
export OPENAI_API_KEY="your_key_here"
(or set it in Windows PowerShell / cmd appropriately)

2. Confirm any configuration variables in the scripts (for example: model names, vector store path, and threshold values). The scripts expect to find the `data/` folder in the repository root.

## How to run

1) Create the memory/vector store (RAG index)
- Purpose: Converts your knowledge base into embeddings and stores them in a vector index that medi_bot.py will use for retrieval.
- Run:
  python create_memory_for_LLM.py

This script will:
- Read the knowledge base / source documents used for retrieval (see the script for the exact source file used).
- Generate embeddings and persist a vector store / index (check the script to see output path; common outputs are a directory like `./vector_store/` or a file such as `vectordb.*`).

2) Run the LLM-based defense pipeline
- Purpose: Executes automated prompts (from your datasets) through the defensive pipeline and logs the results.
- Run:
  python medi_bot.py

This script will:
- Load the vector store created in step 1.
- Load prompts from the CSV files under `data/` (my_Dataset.csv and medicine_related.csv).
- For each prompt, run the evaluation/defense flow using the configured LLM and retrieval steps.
- Use a cosine similarity comparison (threshold) to decide how to treat retrieved context vs. prompts. See the code for exact logic and output location.

## About the cosine similarity threshold

- During development you used `testing.csv` to find a suitable cosine similarity cutoff to separate relevant/benign context from adversarial or irrelevant matches.
- The fitted threshold value is hard-coded in the code (look for a variable named similar to `THRESHOLD`, `COSINE_SIM_THRESHOLD`, or in the section of medi_bot.py that compares cosine similarity). To change the sensitivity, update this value and re-run tests (or re-run the pipeline).

## Data files (quick summary)

- data/my_Dataset.csv — adversarial prompt list (used to test attack-resilience).
- data/medicine_related.csv — benign medical prompts (used to test utility of the pipeline).
- data/testing.csv — used to analyze and choose the cosine similarity threshold during development.

## How to add your published paper (PDF) to the GitHub repository

You mentioned you have a published paper on your laptop and want to show it on GitHub. Here are recommended options (pick one based on size and preference):

1. Add the PDF directly to the repo (small files)
- Copy the PDF into the repository root (for example `paper/your_paper.pdf`).
- Commit and push:
  git add paper/your_paper.pdf
  git commit -m "Add published paper PDF"
  git push origin main

2. Use Git LFS for large PDFs (>100 MB)
- Install Git LFS: https://git-lfs.github.com/
- Track the PDF extension and push:
  git lfs install
  git lfs track "*.pdf"
  git add .gitattributes
  git add paper/your_paper.pdf
  git commit -m "Add paper via LFS"
  git push origin main

3. Attach the PDF to a GitHub Release
- If you prefer not to keep the PDF in the main tree, create a release in the repository (via the Releases UI) and attach the PDF binary to the release.

4. Host externally and link
- Upload the PDF to a repository hosting service (Zenodo will mint a DOI), or your institution's site, then add a link to the README with citation info.

Suggested README entry after uploading:
- Add a short citation block with title, authors, year, DOI/link, and a link to the PDF path in the repo (or the release URL).

Example citation block to add to README:
Paper: "Title of your paper", Author(s), Conference/Journal (Year). PDF: ./paper/your_paper.pdf
(or link to DOI/Zenodo)

## Troubleshooting & tips

- If the vector store is not found, re-run create_memory_for_LLM.py and check its output path.
- If LLM calls fail, verify your API key environment variable is set and you have network access.
- To tune the defense, edit the cosine similarity threshold and run experiments using data/testing.csv to measure false-positive / false-negative tradeoffs.

## License & citation

- Include the correct license for your project (add a LICENSE file if you want to make it open source).
- Add a citation for your paper in this README once you upload it (see "How to add your published paper").

---

If you'd like, I can:
- Add a small requirements.txt based on the code,
- Add a sample config file for API keys/paths,
- Or prepare a short snippet to upload your paper to the repo and add the citation automatically.
