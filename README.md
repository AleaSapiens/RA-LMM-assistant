# RA-LMM-assistant

RA-LMM-assistant is a Streamlit application for predicting whether patients with rheumatoid arthritis (RA) have comorbid low muscle mass (LMM). It combines a trained Random Forest model, SHAP feature attribution, and an LLM-based LangGraph agent workflow to generate a patient-facing explanatory report focused on RA-LMM risk.

> Note: This application is for research and educational use only. It does not provide a medical diagnosis. Users should consult a qualified clinician for medical decisions.

## Features

- Predicts the probability that a rheumatoid arthritis patient has comorbid low muscle mass from five clinical inputs: ANC, ALT, AST, Gender, and BMI.
- Loads a trained scikit-learn Random Forest pipeline from `models/random_forest_best_model.pkl`.
- Computes SHAP values for single-sample feature attribution.
- Uses a LangGraph workflow with RA-LMM prediction, patient-facing report drafting, and self-reflection nodes.
- Calls the DeepSeek-compatible OpenAI API through `langchain-openai` to generate and review the final RA-LMM explanation.
- Provides a Streamlit web interface and Docker deployment support.

## Project Structure

```text
RA_LMM_project/
├── app.py
├── Dockerfile
├── requirements.txt
├── uv.lock
└── models/
    └── random_forest_best_model.pkl
```

## Requirements

- Python 3.12 or later is recommended because `uv.lock` declares `requires-python = ">=3.12"`.
- A DeepSeek API key is required to run the LLM report generation workflow.
- The model file must exist at `models/random_forest_best_model.pkl`, unless `MODEL_PATH` is set.

The main Python dependencies are listed in `requirements.txt`, including:

- streamlit
- pandas
- scikit-learn
- joblib
- shap
- langgraph
- langchain
- langchain-openai
- openai

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Docker Usage

Build the image:

```bash
docker build -t ra-lmm-assistant .
```

Run the container:

```bash
docker run --rm -p 8501:8501 ra-lmm-assistant
```

Then visit:

```text
http://localhost:8501
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` | `models/random_forest_best_model.pkl` | Path to the trained model file. In Docker it is set to `/app/models/random_forest_best_model.pkl`. |

The DeepSeek API key is entered in the Streamlit UI and is not stored by the application.

## Input Fields

| Field | Type | Description |
| --- | --- | --- |
| `ANC` | float | Absolute neutrophil count. |
| `ALT` | float | Alanine aminotransferase. |
| `AST` | float | Aspartate aminotransferase. |
| `Gender` | categorical | `male` is encoded as `1`; `female` is encoded as `0`. |
| `BMI` | float | Body Mass Index. |

## Workflow

1. The Streamlit form collects clinical indicators from a rheumatoid arthritis patient and the DeepSeek API key.
2. The model predicts the probability and risk class for comorbid low muscle mass.
3. SHAP values are calculated for the same input sample to show which variables increased or decreased the RA-LMM prediction.
4. A LangGraph drafting node asks the LLM to generate a structured RA-LMM explanation for the patient.
5. A reflection node reviews whether the report clearly states the RA-LMM probability, explains SHAP evidence, and includes a medical disclaimer.
6. The UI displays low muscle mass probability, prediction, SHAP attribution, and the final RA-LMM-assistant report.

## Notes and Known Issues

- `uv.lock` requires Python `>=3.12`, but the Dockerfile currently uses `python:3.11-slim`. Consider changing the Docker base image to Python 3.12 or aligning the lock file with Python 3.11.
- The saved model was created with scikit-learn 1.8.0. Loading it with another scikit-learn version may raise `InconsistentVersionWarning` and can lead to unreliable results. Use the same scikit-learn version as the model was trained with when possible.
- `requirements.txt` includes several heavy packages that are not used directly by `app.py`, such as `torchvision`, `xgboost`, `catboost`, `seaborn`, and `statsmodels`. Removing unused dependencies can reduce install time and Docker image size.
- `calculate_single_sample_shap()` assumes the model pipeline has named steps `scaler` and `RF`. If the model is replaced, keep the same step names or update the code.
- The threshold is hard-coded as `0.5` for high risk classification. If this threshold was chosen by validation analysis, document it with the model training process.

## Disclaimer

This project provides machine-learning-based estimation of low muscle mass comorbidity risk in rheumatoid arthritis patients and LLM-generated explanations. It should not be used as a standalone clinical decision system. Always interpret outputs with clinical context and professional medical judgment.


