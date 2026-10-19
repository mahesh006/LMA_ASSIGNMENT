# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

This project compares two retrieval pipelines for Retrieval-Augmented Generation (RAG) on one complex legal document, the **EU General Data Protection Regulation (GDPR)**.

- **Pipeline 1 – Baseline:** a bi-encoder retrieves the **Top 10** chunks by cosine similarity, and the **Top 3** go to the LLM to produce **Answer 1**.
- **Pipeline 2 – Re-Ranking:** a cross-encoder re-scores the same Top 10, and the new **Top 3** go to the LLM to produce **Answer 2**, using the identical prompt.

Both pipelines use exactly the same chunks, the same LLM and the same prompt, so any difference in the answers comes only from the retrieved context.

```
                     ┌──────────────────────────────┐
  Query ──► Bi-encoder + FAISS ──► Top 10 chunks ───┤
                     └──────────────────────────────┘
                                   │
           ┌───────────────────────┴───────────────────────┐
           ▼                                               ▼
   Pipeline 1: take Top 3                   Pipeline 2: Cross-encoder re-scores
   by cosine similarity                     all 10 (query, chunk) pairs → Top 3
           │                                               │
           ▼                                               ▼
     LLM → Answer 1                                  LLM → Answer 2
```

---

## Results

Ten queries were written to stress the baseline: keyword traps, exceptions and negation. Each query has a *gold chunk*, the chunk that actually contains the answer.

| | Baseline | Re-ranked |
|---|---|---|
| Gold chunk in Top 3 | **7 / 10** | **10 / 10** |
| Answers correct / partial / wrong (judged by hand) | 4 / 2 / 4 | **6 / 3 / 1** |
| Retrieval latency per query | 8.02 ms | 24.19 ms (+16.17 ms re-ranking) |
| LLM generation (average) | 4.38 s | 3.86 s |

### Queries restored by re-ranking
| Query | Failure type | Gold chunk: baseline → re-ranked | Baseline #1 chunk (distractor) |
|---|---|---|---|
| Q6: When can a request for erasure be refused? | Exception | Art. 17(3), **#6 → #3** | general erasure text |
| Q7: Can a public authority rely on legitimate interests? | Exception | Recital 47, **#4 → #1** | public-interest tasks |
| Q8: When is processing health data *not* prohibited? | Negation | Recital 52, **#7 → #1** (best gold #5 → #1) | exceptions to the duty to *inform* |

### Effect on the answers
- **Q8:** the baseline answer misstated the rule, using exceptions from a different article. The re-ranked answer is correct.
- **Q5:** the baseline answer hallucinated an extra "public communication" condition. The re-ranked answer is correct.
- **Q7:** the baseline gave no answer. The re-ranked answer is partially correct.
- **No query got worse.** Where the baseline already had the gold chunk at #1, re-ranking kept it in the Top 3.

### Where re-ranking was not enough
- **Q4:** the exemption in Art. 30(5) moved only from #9 to #4, so it never reached the LLM. Both answers were wrong.
- **Q3:** Art. 22 moved from #7 to #4. Both answers were partial.
- **Q6:** the right article reached the LLM, but both answers stayed incomplete. Re-ranking fixes *what* the LLM sees, not *how well it uses it*.
- **Q1–Q3:** these designed keyword-trap queries did not fool the bi-encoder. The keyword-trap effect appears instead in the distractor chunks of Q6–Q8.

### Scaling
For one query, the cross-encoder over all 458 chunks took **349 ms**, against **23 ms** for the two-stage pipeline (about 0.76 ms per pair). Extrapolated, that is ≈ 1.3 min per query at 100k chunks and ≈ 12.7 min at 1M chunks. This is why the cross-encoder is only applied to the Top 10.

The full analysis is in [`Report.pdf`](Report.pdf). The LaTeX source is in [`report/report.tex`](report/report.tex).

---

## Repository structure

```
LMA_ASSIGNMENT/
├── RAG_Baseline_vs_Reranking.ipynb   # all code, saved with outputs
├── Report.pdf                        # analysis report
├── report/
│   └── report.tex                    # LaTeX source of the report
├── requirements.txt                  # Python dependencies
├── data/
│   └── gdpr.pdf                      # source document (GDPR, EUR-Lex)
└── results/                          # written by the notebook
    ├── summary.csv                   # gold-chunk rank per query: baseline vs re-ranked
    ├── rank_shifts.csv               # full Top-10 rank-shift table for every query
    ├── answers.md                    # Answer 1 vs Answer 2 for every query
    └── latency.csv                   # latency table
```

---

## Models and settings

| Component | Choice |
|---|---|
| Document | GDPR, Regulation (EU) 2016/679, 88 pages |
| Chunking | 150-word windows with 30-word overlap (458 chunks) |
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS `IndexFlatIP` on L2-normalised vectors (inner product = cosine similarity) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `Qwen/Qwen2.5-7B-Instruct`, float16, greedy decoding |
| Top K / Top N | 10 retrieved, 3 passed to the LLM |

All settings are in the configuration cell (Section 1) of the notebook.

---

## How to run

### 1. Requirements
- Python **3.10+**
- An NVIDIA GPU. The 7B LLM needs about **15 GB of GPU memory** in float16; `device_map="auto"` splits it across GPUs if one is not enough. I used 4 × RTX 2080 Ti (11 GB each). 2 × T4 on Kaggle also works.
- About **20 GB of free disk space** for the model downloads (Hugging Face cache).

### 2. Clone the repository
```bash
git clone https://github.com/mahesh006/LMA_ASSIGNMENT.git
cd LMA_ASSIGNMENT
```

### 3. Create an environment and install packages
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# PyTorch with CUDA (pick the CUDA version that matches your driver, see pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# everything else
pip install -r requirements.txt
```

With conda instead:
```bash
conda create -n rag python=3.10 -y
conda activate rag
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Check that the GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### 4. Run the notebook
```bash
jupyter notebook RAG_Baseline_vs_Reranking.ipynb
```
Then choose **Kernel → Restart & Run All**.

- On the first run, the three models download automatically from Hugging Face (the LLM is about 15 GB).
- After that, a full run takes a few minutes. Generating the 20 answers takes about 80 s of that (about 4 s per answer).
- The results are written to the `results/` folder.

### Running on Kaggle (optional)
1. Create a new notebook and import `RAG_Baseline_vs_Reranking.ipynb`.
2. In the settings panel, set **Accelerator → GPU T4 x2** and **Internet → On**.
3. Upload `data/gdpr.pdf` with **Add Input → Upload → New Dataset**.
4. Add a first cell with `!pip install -q pypdf faiss-cpu`.
5. Set `PDF_PATH` in the config cell to the uploaded file's path, e.g. `/kaggle/input/gdpr/gdpr.pdf`.
6. Choose **Run All**.

### Troubleshooting
| Problem | Fix |
|---|---|
| `CUDA out of memory` when loading the LLM | Use more GPUs, or set `LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"` (about 7 GB) |
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` |
| `FileNotFoundError: data/gdpr.pdf` | Run the notebook from the repository root, or set `PDF_PATH` in the config cell |
| A Section 8 warning: "phrase not found" | You are using a different GDPR PDF. Use the one in `data/` |

---

## Notebook walkthrough

| Section | What it does |
|---|---|
| 1. Configuration | model names, chunk size, Top K / Top N |
| 2. Load PDF | extract text with `pypdf`, remove page headers, fix hyphenation |
| 3. Chunking | 150-word windows with 30-word overlap |
| 4. Indexing | bi-encoder embeddings into a FAISS index |
| 5–6. Models | load the cross-encoder and the LLM; define the shared prompt |
| 7. Pipelines | `baseline_retrieve()` and `rerank()` |
| 8. Test queries | 10 failure-mode queries (keyword trap, exception, negation) with gold phrases |
| 9. Run | both pipelines on every query: rankings, answers, timings |
| 10. Rank shifts | summary table, per-query Top-10 tables, buried-chunk cases |
| 11. Answers | gold text, Answer 1 and Answer 2 side by side |
| 12. Latency | retrieval and re-ranking timed over 10 queries × 20 repeats |
| 13. Scaling | cross-encoder on all chunks vs two-stage, extrapolated to 1M chunks |
| 14. Save | write the CSV and Markdown files to `results/` |

### How retrieval quality is measured
Each test query has one or more **gold phrases**: verbatim GDPR text that contains the correct answer, from the Article and/or its matching Recital. A chunk containing a gold phrase is a **gold chunk**. Its rank is tracked in both pipelines and labelled with one of these outcomes:
- **Restored:** outside the baseline Top 3, inside the re-ranked Top 3
- **Both OK:** inside the Top 3 in both pipelines
- **Degraded:** pushed out of the Top 3 by re-ranking
- **Not recoverable:** not in the Top 10 at all

---

## Document source
GDPR, Regulation (EU) 2016/679, from EUR-Lex: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679
