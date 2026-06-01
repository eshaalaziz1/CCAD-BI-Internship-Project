# Tumor Board AI Starter

Local prototype for using Ollama + MedGemma 1.5 4B to summarize synthetic oncology patients for MDT/tumor board review.

GitHub repo: [eshaalaziz1/CCAD-BI-Internship-Project](https://github.com/eshaalaziz1/CCAD-BI-Internship-Project)

## Setup

Keep Ollama running in one terminal:

```bash
ollama serve
ollama pull medgemma1.5
```

In another terminal:

```bash
cd tumor-board-ai-starter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Streamlit app (local)

```bash
streamlit run app.py
```

Ollama chat uses `temperature: 0.2`, `num_predict: 700`, and `keep_alive: 30m`. Prompt text lives in `prompts/tumor_board_summary.txt` (`PROMPT_VERSION` in code). Summaries are cached by patient id, record content, and cache-bust token.

**App features:** side-by-side record/summary, cache/regenerate/clear, Markdown export, streaming on **Regenerate**.

**Batch evaluation:**

```bash
python scripts/run_batch_eval.py
```

See [EVAL.md](EVAL.md) for the review rubric.

**M1 MacBook Air:** MedGemma 1.5 4B via Ollama is safe for local prototyping on 8 GB unified memory; close other heavy apps during generation. Use a GPU server when you need always-on or multi-user deployment.

### Jupyter notebook

```bash
pip install notebook
jupyter notebook
```

Open `notebooks/01_medgemma_testing.ipynb`.

## Deploy on Streamlit Community Cloud

1. Push this project to `CCAD-BI-Internship-Project` on GitHub (see below).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. **New app** → repository `eshaalaziz1/CCAD-BI-Internship-Project`, branch `main`, main file path `app.py`.
4. Deploy.

**Important:** Streamlit Cloud runs in the cloud and cannot reach **Ollama on your laptop**. The UI will load, but **Generate tumor board summary** only works when Ollama is reachable (local `streamlit run app.py`, or a server where you run `ollama serve` and set `OLLAMA_HOST` if needed).

## Push to GitHub

From this folder (first time):

```bash
git init
git remote add origin https://github.com/eshaalaziz1/CCAD-BI-Internship-Project.git
git add .
git commit -m "Add Streamlit tumor board app and synthetic patient data"
git branch -M main
git push -u origin main
```

## Notes

- Use synthetic data only for early prototyping.
- Do not commit real PHI/medical records.
- Do not commit model weights.
- Validate outputs with clinicians before any real-world use.
