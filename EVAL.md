# Evaluation guide

Synthetic-only prototype. Use this checklist before changing prompts or model settings.

## Quick batch run

```bash
ollama serve   # separate terminal
source .venv/bin/activate
python scripts/run_batch_eval.py
```

Outputs land in `outputs/eval_<timestamp>.jsonl` with per-patient latency, summary text, and errors.

## Manual review rubric

For each patient, score **Yes / Partial / No**:

| Criterion | Question |
|-----------|----------|
| Factual | Does every stated fact appear in the patient record? |
| No invention | Are unknown/pending items called out instead of filled in? |
| Structure | Are all five sections present and on-topic? |
| MDT value | Are discussion questions useful for a tumor board? |
| Safety | Are treatment lines phrased as possibilities, not orders? |

## Edge cases in `sample_patients.csv`

| Patient | What to test |
|---------|----------------|
| P001 | ALK unknown; pending brain MRI |
| P005 | BRCA unknown |
| P006 | ECOG 2, comorbid lung disease |
| P008 | Missing outside pathology details |
| P009 | Molecular profiling pending |
| P010 | MMR/p53 pending after surgery |

## Comparing prompt versions

1. Note `PROMPT_VERSION` in `src/summarizer.py`.
2. Run batch eval → save JSONL with version in filename.
3. Diff summaries for the same `patient_id` across runs.
4. Record reviewer notes in your internship report.

## Limitations (document in demos)

- Local Ollama only; not validated for clinical use.
- No RAG over guidelines or imaging reports.
- CSV is structured text, not full EHR documents.
- Streamlit Community Cloud cannot reach laptop Ollama without a remote host.
