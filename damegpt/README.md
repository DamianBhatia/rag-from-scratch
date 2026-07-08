# DameGPT

A simple ChatGPT-style UI built inside the `damegpt` folder.

## Run locally

1. Install dependencies:
   ```bash
   pip install flask ollama
   ```

2. Start the app from the repository root:
   ```bash
   python damegpt/app.py
   ```

3. Open http://localhost:5000 in your browser.

## Notes

- The app loads knowledge from `cat-facts.txt` in the repository root.
- It uses the local `ollama` package to embed queries and generate chat responses.
