# Deploy / Demo

This folder contains a lightweight FastAPI deployment that serves:

- `/predict` - POST image file to get defect detection JSON + mask overlay.
- `/` - an accessible single-page demo (`static/index.html`) for manual testing.

Quick start

```bash
cd deploy
python3 -m venv venv && source venv/bin/activate
pip install -r deployrequirements.txt
uvicorn deployapp:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/
```

Docker

```bash
cd deploy
docker build -t defect-deploy .
docker run -p 8000:8000 defect-deploy
```

Accessibility

- The demo uses semantic HTML and ARIA live regions to announce results.
- The overlay PNG is embedded in the JSON response as a `mask_image` data URL.
- Use Lighthouse / Pa11y or screen readers (VoiceOver/NVDA) to validate accessibility.
