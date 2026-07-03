# India Runs Data Challenge

A Flask-based candidate ranking pipeline that automatically ingests recruiting datasets, scores and ranks candidates using a LightGBM ranker, generates human-readable hiring reasoning for each top candidate, and displays results in a live, auto-refreshing dashboard.

## Features

- **Automated file watching** — Drop a `.csv`, `.xlsx`, `.xls`, `.json`, or `.jsonl` file into the `data/` folder and the pipeline picks it up and processes it automatically (via `watchdog`).
- **Candidate ranking** — Uses a `LightGBMRanker` trained on engineered features (recency, recruiter responsiveness, offer acceptance, GitHub activity, profile completeness, skill match) to rank candidates.
- **AI-generated reasoning** — Produces a short hiring rationale for each top candidate using a FLAN-T5 (Transformers) model when available, with a robust template-based fallback when no GPU/model is present.
- **Live dashboard** — A built-in web UI (Tailwind-styled) shows:
  - A 3D podium view of the top 3 candidates
  - A searchable, paginated candidate table
  - Real-time pipeline/model/watcher status
  - A live status/log feed
- **CSV export** — Download the ranked results as `submission.csv` directly from the dashboard.
- **Resilient model loading** — Automatically migrates legacy model files into a consolidated `model.pkl`, or trains a fresh ranker if none exists.

## Tech Stack

- **Backend:** Flask, pandas, NumPy, LightGBM, scikit-learn, joblib
- **AI Reasoning (optional):** PyTorch, Hugging Face Transformers (FLAN-T5)
- **File watching:** watchdog
- **Frontend:** Server-rendered HTML with Tailwind CSS (CDN), vanilla JS
- **Containerization:** Docker

## Project Structure

```
.
├── app.py               # Main Flask application and processing pipeline
├── Dockerfile            # Container build definition
├── requirements.txt       # Python dependencies
├── model.pkl             # Trained LightGBM ranker (auto-created/updated)
├── data/                 # Drop input datasets here (auto-processed)
├── result/               # Output submission.csv is written here
└── .dockerignore
```

## Getting Started

There are three ways to run this project: **pulling the prebuilt Docker image**, **building the Docker image yourself from source**, or **running it manually with Python**.

---

### Option 1: Pull and run the prebuilt Docker image (fastest)

**1. Pull the image:**
```bash
docker pull vansh940/india_run:latest
```

**2. Run the container:**
```bash
docker run -d -p 5000:5000 \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/result:/app/result \
  --name india-runs \
  vansh940/india_run:latest
```

**3. Open the dashboard:**
Visit **http://localhost:5000** in your browser.

**4. Check logs (optional, useful for debugging):**
```bash
docker logs -f india-runs
```

**5. Stop / remove the container when done:**
```bash
docker rm -f india-runs
```

> **Note (Windows PowerShell):** `${PWD}` works natively. If you're using `cmd.exe`, replace `${PWD}` with `%cd%`.

---

### Option 2: Build the Docker image yourself from source

If you've cloned this repo and want to build the image locally instead of pulling it:

```bash
# From the project root (where the Dockerfile lives)
docker build -t india_run:latest .

# Run it the same way as Option 1
docker run -d -p 5000:5000 \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/result:/app/result \
  --name india-runs \
  india_run:latest
```

For a completely clean rebuild (ignoring any cached layers):
```bash
docker build --no-cache -t india_run:latest .
```

---

### Option 3: Run manually with Python (no Docker)

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the app:**
```bash
python app.py
```

The server starts on `http://0.0.0.0:5000` by default. You can override the port with the `PORT` environment variable:
```bash
PORT=8080 python app.py
```

**3. Open the dashboard:**
Visit **http://localhost:5000** (or whichever port you set) in your browser.

> **Note:** Running manually requires `libgomp` (needed by LightGBM) to already be present on your system. This is preinstalled on most desktop OSes; on minimal Linux environments you may need to install it separately (e.g. `apt-get install libgomp1`).

---

## Usage

1. Place a candidate dataset file (`.csv`, `.xlsx`, `.xls`, `.json`, or `.jsonl`) into the `data/` folder.
2. The background watcher detects the new file and automatically:
   - Parses and engineers features from the data
   - Ranks candidates using the LightGBM model (training one if none exists yet)
   - Generates reasoning for the top 100 candidates
   - Writes results to `result/submission.csv`
3. View live results on the dashboard at `http://localhost:5000`.
4. Download the ranked output using the **Download submission.csv** button.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Dashboard UI |
| `POST` | `/upload` | Manually upload a dataset file |
| `GET`  | `/api/check-status` | Current pipeline/model/watcher status and ranked rows |
| `GET`  | `/preview` | Preview the top 10 ranked candidates |
| `GET`  | `/download/submission.csv` | Download the ranked results |
| `POST` | `/clear_logs` | Clear the status log feed |
| `POST` | `/api/delete_result` | Delete the current `submission.csv` |
| `POST` | `/api/delete_input` | Delete a specific input file from `data/` |
| `POST` | `/api/reset_all` | Reset pipeline state (clears results, logs, processed file tracking) |

## Notes

- If PyTorch/Transformers are unavailable, fail to load, or the FLAN-T5 model can't be downloaded, the app automatically falls back to template-based reasoning generation — no functionality is lost, only the reasoning becomes rule-based instead of model-generated.
- GPU acceleration is used automatically if available (`torch.cuda.is_available()`); otherwise the pipeline runs on CPU.
- On first run, if the FLAN-T5 model isn't cached locally, it will be downloaded from Hugging Face Hub, which may take a few minutes depending on your connection.