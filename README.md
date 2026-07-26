# Resume RAG Assistant

A Streamlit application that indexes uploaded PDF, DOCX, and TXT resumes with
OpenAI embeddings and answers questions using retrieved resume excerpts.

## Run locally

Use Python 3.12 so the local environment matches Streamlit Community Cloud:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Either enter an OpenAI API key in the app sidebar, set `OPENAI_API_KEY` in a
local `.env` file, or create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-key"
```

Do not commit either secrets file. Both are excluded by `.gitignore`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and push this project. Do not add `.env`,
   `.streamlit/secrets.toml`, `.venv`, or the `data` folder.
2. Sign in at <https://share.streamlit.io> with GitHub and create an app.
3. Select the repository and branch, and set the entrypoint to `app.py`.
4. In **Advanced settings**, select Python 3.12 and add:

   ```toml
   OPENAI_API_KEY = "your-key"
   ```

5. Deploy. Dependency installation is driven by `requirements.txt`.

The Streamlit hosting tier can be free, but OpenAI API calls are billed to the
configured OpenAI account. Uploaded resumes are held in the app session; the
application does not deliberately save them to the repository or a database.

## Limitations

- Legacy `.doc` files are not supported; convert them to `.docx` or PDF.
- Image-only/scanned PDFs require OCR before upload.
- FAISS indexes are in-memory and are rebuilt when the session restarts.
- Anyone who can access a publicly deployed app can consume the configured API
  key indirectly, so add access controls or require users to enter their own key
  before sharing broadly.
