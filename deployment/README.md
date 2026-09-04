# deployment/

The deployed proof-of-concept (Phase 3). Plan: a **Streamlit** app deployed on **Streamlit Cloud**
that loads a trained model and predicts on held-out historical orders (simulating new orders).

Planned layout:
- `app.py` — Streamlit entry point.
- `model.joblib` — trained pipeline (or loaded from a release/artifact).
- `requirements.txt` — deployment-only pins (can reference the root file).

Requirements to meet: live URL, demonstrates at least one model, clear usage instructions,
no hardcoded credentials, basic input validation / error handling.
