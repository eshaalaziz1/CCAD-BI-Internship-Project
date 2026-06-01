import ollama

MODEL = "medgemma1.5"

def summarize_patient(patient_data: str, model: str = MODEL) -> str:
    prompt = f"""
You are assisting a multidisciplinary oncology tumor board.

Use ONLY the patient information provided.
Do not invent missing biomarkers, imaging findings, pathology, staging, or treatment history.

Return:
1. One-line case summary
2. Key clinical facts
3. Missing or unclear information
4. MDT discussion questions
5. Treatment considerations, phrased as possibilities, not final recommendations

Patient data:
{patient_data}
"""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
