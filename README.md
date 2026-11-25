# mlops-rides 🚲

End-to-end MLOps pipeline for a **deep learning model that classifies rides into 5 categories** from IMU sensor data, with experiment tracking, S3 model storage, a REST API, CI/CD and deployment on Kubernetes.

---

## Project goal

### The problem
Each ride is recorded by **IMU sensors** (accelerometer / gyroscope) as a time series. The goal is to **classify each ride's riding behaviour** into one of 5 categories, and above all to **catch dangerous rides (*Reckless*)**.

### The data — `GBG500.parquet`
500 rides in Gothenburg, each with a time index and its IMU readings. Labels are in `GBG500_labels.csv`.

| Class | Rides | Share | Note |
|---|---|---|---|
| **Safe** | 416 | 83.2 % | Normal riding |
| **Safe Overall** | 38 | 7.6 % | Borderline case |
| **Bad Weather** | 22 | 4.4 % | Borderline case |
| **Tandem** | 18 | 3.6 % | Borderline case |
| **Reckless** | 6 | 1.2 % | 🎯 Most important class to detect |

The three borderline classes look like abnormal riding (weather, two riders, rougher riding) but are not dangerous. The model must separate them from *Reckless*.

> `SRV3490.parquet` (3,490 rides, fall detection) is not used in this project.

### Model input — design choice
The project brief mentions an API that receives an *image*. Our data is **IMU time series, not images**, so the API receives a **ride as a sequence of sensor readings (JSON)**. This keeps the raw signal and avoids an artificial image conversion.

### Supervised training — design choice
The data description says the labels are *used only for evaluation*, which suggests an anomaly-detection setup (learn normal rides, flag the unusual ones). We train a **supervised 5-class classifier** instead:
- the three borderline classes are *meant* to look abnormal without being dangerous, so an anomaly detector trained without labels would flag them as well; the labels are what teach the model to separate them from *Reckless*;
- the API contract returns a class and 5 probabilities, which an anomaly score cannot give;
- the project grades the MLOps infrastructure, and the model only has to beat chance.

The labels never leak into evaluation: test rides are held out before training and only scored at the end.

### How we measure success
The model only needs to **beat chance**: infrastructure is what is graded. But with 83 % *Safe*, a model that always answers "Safe" gets 83 % accuracy while detecting nothing. We therefore track:
- **macro F1** across the 5 classes;
- **recall on *Reckless***;
- **false alarm rate** on *Safe Overall*, *Bad Weather* and *Tandem*.

**Baseline to beat:** always predict the majority class.
Because *Reckless* has only 6 rides, rides are split **by ride** into train / validation / test (64 / 16 / 20 %), **stratified on the label** so every split keeps every class, and each ride is cut into **fixed-size windows** to get more training samples. The validation set picks the best epoch; the test set is only used for the final score and the evaluation gate. We do not run cross-validation: with a deep model and a registry entry per run, one fixed split keeps the pipeline simple and its results comparable from run to run.

---

##  Architecture

```
data (RustFS + DVC) → training (PyTorch) → tracking & registry (MLflow, artifacts on RustFS)
      → API (FastAPI) → Docker image (Docker Hub) → Kubernetes ← CI/CD GitHub Actions (self-hosted runner)
                                       ↑
                                 demo web app
```

| Environment | Where | Services |
|---|---|---|
| **dev** | local machine, `docker compose` | RustFS, MLflow, Postgres, API, web app |
| **prod** | local Kubernetes (**Docker Desktop**) | **all services**: RustFS, MLflow, Postgres, API, web app (manifests in `k8s/`) |

---

## ✅ Requirements coverage

| # | Requirement | Where |
|---|---|---|
| 1 | Extract data | `src/rides/data.py` |
| 2 | Deep learning classification model (5 classes) | `src/rides/model.py`, `src/rides/train.py` (PyTorch) |
| 3 | Model stored on S3 (RustFS, S3-compatible) | MLflow artifacts → bucket `mlflow` |
| 4 | Track models and experiments | MLflow, model `rides-classifier` |
| 5 | Prediction API | `src/rides/api/` (FastAPI, Swagger at `/docs`) |
| 6 | Dockerized API on Kubernetes via GitHub Actions | `docker/`, `k8s/`, `.github/workflows/` |
| 7 | Versioned and documented on GitHub | this repo |
| Bonus | Retraining pipeline | `.github/workflows/train.yml` |
| Bonus | Monitoring | `monitoring/` (Prometheus + Grafana) |
| Bonus | Feature store | Feast (optional) |


---

## 👥 Team and ownership

| # | Member | Scope | Folders / files |
|---|---|---|---|
| 1 | @Andrianrjs | Data + model | `src/rides/data.py`, `model.py`, `train.py` |
| 2 | @mnegron-cloud| MLflow + DVC pipeline | `src/rides/evaluate.py`, `dvc.yaml`, `params.yaml` |
| 3 | @sjoshi-rgb | API + web app | `src/rides/api/`, `webapp/` |
| 4 | @lcaquelin | Docker + Kubernetes | `docker/`, `docker-compose.yml`, `k8s/` |
| 5 | @clemencemarco-prog | CI/CD + tests + bonus | `.github/workflows/`, `tests/`, `monitoring/` |

**Everyone**: final README, screenshots, defence preparation.

### 1 · Data + model — @Andrianrjs
- [x] Explore `GBG500.parquet`: IMU columns, sampling rate, ride lengths, class imbalance
- [x] `data.py`: loading, join with labels, fixed-size windowing, normalisation, stratified split
- [x] `model.py`: small PyTorch network (1D CNN or LSTM)
- [x] `train.py`: training + `mlflow.log_params / log_metrics` + register the model as **`rides-classifier`**
- [x] Handle class imbalance (class weights or oversampling)
- **Hands over:** the exact model input format: window length **T**, list of the **C** sensor columns, normalisation stats

### 2 · MLflow + pipeline — @mnegron-cloud
- [x] `evaluate.py`: macro F1, Reckless recall, confusion matrix logged to MLflow; promote to **`@staging`** if better than baseline
- [x] DVC: `dvc init`, RustFS remote (bucket `dvc`), `dvc add data/raw/GBG500.parquet`, `dvc push`
- [x] `dvc.yaml`: `prepare → train → evaluate` DAG (the required pipeline DAG)
- [x] `params.yaml`: hyperparameters
- **Hands over:** a `models:/rides-classifier@staging` model in MLflow

### 3 · API + web app — @sjoshi-rgb
- [x] `src/rides/api/`: `/health`, `POST /v1/predict`, Swagger docs (`description`, `examples`)
- [x] Adapt `schemas.py` to the input format from @Andrianrjs
- [x] `webapp/app.py` (Streamlit): pick or upload a ride, call the API, show the class and probabilities
- 💡 Until the real model exists: test with a dummy model returning random probabilities
- **Hands over:** an API running with `uv run uvicorn rides.api.main:app`

### 4 · Docker + Kubernetes — @lcaquelin
- [x] Clean up the Docker setup: pinned image versions, Postgres and MLflow healthchecks, MinIO services renamed to RustFS, outdated `docker/mlflow/docker-compose.yml` removed
- [x] `docker/api.Dockerfile` (CPU-only torch, non-root user, healthcheck) and `.dockerignore`
- [x] `docker/webapp.Dockerfile` (only the `webapp` dependency group: no torch)
- [x] API and web app added to `docker-compose.yml` (dev)
- [x] Kubernetes enabled in Docker Desktop (kubeadm, 1 node)
- [x] `k8s/`: RustFS, Postgres, MLflow, API and web app on the cluster (prod = all services), with readiness / liveness probes and resource limits
- [x] Images pushed to Docker Hub, multi-architecture (`linux/amd64` + `linux/arm64`)
- **Hands over:** `docker compose up -d` starts dev · `kubectl apply -f k8s/` starts prod

### 5 · CI/CD + tests + bonus — @clemencemarco-prog
- [x] `ci.yml`: ruff lint + unit and integration tests on every push and PR (GitHub-hosted runner)
- [x] `cd.yml`: build → e2e test of the image → push to Docker Hub (multi-arch) → deploy to Kubernetes → smoke test, automatic rollback (`runs-on: self-hosted`)
- [x] `tests/unit`, `tests/integration` (API + real MLflow model with TestClient), `tests/e2e` (API in Docker → request → valid response)
- [x] Bonus: `train.yml` retraining workflow (pull data → retrain → gate → redeploy if promoted)
- [x] Bonus: Prometheus + Grafana (API latency, request count, input and prediction distribution, alerts)
- [ ] Self-hosted runner registered (Settings → Actions → Runners) on the machine running the prod cluster
- [ ] Bonus: feature store (Feast), not planned
- **Hands over:** a push to `main` that passes CI is tested, published and deployed without manual steps

---

##  Contracts between parts
Fixed now so everyone can work in parallel. **Do not change without telling the team.**

| Item | Valeur |
| --- | --- |
| T | 100 pas (10 s à 100 ms). Le modèle accepte aussi un trajet entier, T ≥ 10 |
| C | 6 : `ax, ay, az, rx, ry, rz`, dans cet ordre |
| Normalisation | **à l'intérieur du modèle**, l'API envoie les valeurs brutes |
| Noms des classes | `Safe, Safe overall, Bad weather, Tandem, Reckless` |
| MLflow model name | `rides-classifier` |
| Alias used by the API | `@staging` |
| Model input | `float32`, shape `(batch, T, C)` *(T and C set by @Andrianrjs)* |
| Model output | probabilities, shape `(batch, 5)`, order: Safe, Safe Overall, Bad Weather, Tandem, Reckless |
| API route | `POST /v1/predict` |
| RustFS buckets | `mlflow` (artifacts), `dvc` (data) |


---

## 🛠️ Getting started

```bash
git clone https://github.com/Andrianrjs/mlops-rides.git && cd mlops-rides
uv sync
cp .env.example .env
docker compose up -d     # MLflow → http://localhost:5001 · RustFS console → http://localhost:9001
uv run dvc remote modify --local storage access_key_id minio            # RustFS credentials,
uv run dvc remote modify --local storage secret_access_key minio-secret # kept out of git (.dvc/config.local)
uv run dvc pull data/raw # raw data only (first time: see "Data with DVC" below)
uv run --env-file .env dvc repro             # prepare → train → evaluate, sets @staging
uv run uvicorn rides.api.main:app --reload   # Swagger → http://localhost:8000/docs
```

---

## 🐳 Docker images

| Image | Docker Hub | Built from |
|---|---|---|
| API | [louisec13012/rides-api](https://hub.docker.com/r/louisec13012/rides-api) | `docker/api.Dockerfile` |
| MLflow server | [louisec13012/rides-mlflow](https://hub.docker.com/r/louisec13012/rides-mlflow) | `docker/mlflow/Dockerfile` |
| Web app | [louisec13012/rides-webapp](https://hub.docker.com/r/louisec13012/rides-webapp) | `docker/webapp.Dockerfile` |

Images on Docker Hub are **multi-architecture** (`linux/amd64` and `linux/arm64`): Docker pulls the
native one on Windows/Linux PCs and on Apple Silicon Macs. `docker compose build` builds them locally
under the same names, for your machine's architecture only, so dev and prod run the same images.

**Publish a new version** with `buildx`, never with `docker compose build` + `docker push`
(that would replace the multi-architecture image with a single-architecture one):

1. Pick a **new tag** (e.g. `0.2.0`) and update it in `docker-compose.yml` and `k8s/`. Never reuse a
   published tag: machines that already pulled it would keep the old image.
2. From the repo root:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -f docker/api.Dockerfile -t louisec13012/rides-api:<tag> --push .
docker buildx build --platform linux/amd64,linux/arm64 -t louisec13012/rides-mlflow:<tag> --push docker/mlflow
docker buildx build --platform linux/amd64,linux/arm64 -f docker/webapp.Dockerfile -t louisec13012/rides-webapp:<tag> --push .
```

---

## 🧪 Run in dev (docker compose)

Every service runs in Docker on your machine; you run training from your terminal.
Prerequisite: the data in `data/raw/` (see Getting started).

```bash
cp .env.example .env                          # Windows: Copy-Item .env.example .env
docker compose up -d --build --remove-orphans # RustFS, Postgres, MLflow, API, web app
uv run --env-file .env dvc repro              # trains and puts a model in @staging
docker compose restart api                    # the API loads the @staging model at startup
curl http://localhost:8000/health             # Windows: curl.exe → {"status":"ok", ...}
```

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5001 |
| API (Swagger) | http://localhost:8000/docs |
| RustFS console | http://localhost:9001 |
| Web app | http://localhost:8501 |

`--remove-orphans` stops the old `minio` / `minio-init` containers if you started the stack
before they were renamed; your data is kept (the volume is still named `minio-data`).

---

## 🚀 Deploy to prod (Kubernetes on Docker Desktop)

Prod runs **every service** on the local Kubernetes cluster, in the `rides` namespace,
on ports that do not clash with dev, so both can run at the same time.

**Once:** Docker Desktop → Settings → Kubernetes → *Enable Kubernetes* (kubeadm), then:

```bash
kubectl config use-context docker-desktop
kubectl get nodes                             # docker-desktop   Ready
```

**Deploy:**

```bash
kubectl apply -f k8s/00-namespace.yaml
# credentials never go to git; safe to re-run (one line, so it works in bash and PowerShell)
kubectl create secret generic rides-secrets -n rides --from-env-file=.env --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/
kubectl get pods -n rides -w                  # wait until all are Running 1/1 (rustfs-init: Completed)
```

The API stays `0/1` at first: the prod MLflow starts empty, so there is no model to serve yet
(its readiness probe keeps traffic away until there is one).

**Register the first prod model** (needs `data/processed/windows.npz`, i.e. one dev `dvc repro`):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5002   # Windows: $env:MLFLOW_TRACKING_URI = "http://localhost:5002"
uv run python -m rides.train
uv run python -m rides.evaluate                    # → PROMOTED to @staging
unset MLFLOW_TRACKING_URI                          # Windows: Remove-Item Env:MLFLOW_TRACKING_URI
git checkout -- reports/                           # these reports describe prod, not the commit
uv run dvc checkout                                # restores reports/train_info.json for dev
kubectl rollout restart deployment/api -n rides    # the API loads the new @staging model
curl http://localhost:8080/health                  # Windows: curl.exe
```

| Service | URL |
|---|---|
| MLflow UI (prod) | http://localhost:5002 |
| API (prod, Swagger) | http://localhost:8080/docs |
| Web app (prod) | http://localhost:8502 |

**Remove everything:** `kubectl delete namespace rides` (this also deletes the prod data).

### Design choices (part 4)
- **CPU-only torch in the API image.** On Linux, the default torch build pulls several GB of NVIDIA CUDA libraries that are useless without a GPU. The Dockerfile installs every dependency pinned by `uv.lock` except torch, then torch at the same version from the PyTorch CPU index: about 490 MB to download instead of several GB.
- **Readiness vs liveness.** The API's `/health` answers 503 until a model is loaded. It drives the **readiness** probe (no traffic before a model is there); the **liveness** probe only checks the port answers, so a missing model does not cause a restart loop.
- **MLflow serves the model files** (`--serve-artifacts`): the API needs the MLflow URL only, no storage credentials.
- **MLflow host check.** Since MLflow 3.5, the server rejects unknown Host headers (DNS rebinding protection). `MLFLOW_SERVER_ALLOWED_HOSTS` allows `localhost:*` and `mlflow:5000` (the API inside Docker or Kubernetes); the Kubernetes probes call `localhost` for the same reason.
- **2 MLflow workers.** The default (4) went over the 3 GB memory limit on the cluster (OOMKilled).
- **Multi-architecture images** (`linux/amd64`, `linux/arm64`), built with `docker buildx`: the team works on both Windows PCs and Apple Silicon Macs, and an amd64-only image would run emulated (slow, and numeric libraries like torch can crash under emulation).
- **Secrets** come from `.env` (dev) or a Kubernetes Secret created from it (prod); no credential is committed or baked into an image.
- **Known limit:** `.env.example` holds default development credentials, and everyone copies it. Fine for services reachable only from your machine; a real deployment would set its own values and keep them in a secret manager.
- **Plain manifests rather than Helm:** the same images and settings as in dev, one file per service, easy to read and explain.

---

## ✅ Tests

```bash
uv run pytest                                  # unit + integration (e2e skipped: no API running)
E2E_BASE_URL=http://localhost:8000 uv run pytest tests/e2e   # e2e against a running API (dev compose)
```

| Level | Folder | What it checks | Needs |
|---|---|---|---|
| Unit | `tests/unit/` | ride split, windowing, scaler, model shapes and normalisation, metrics, evaluation gate, contracts between parts | nothing |
| Integration | `tests/integration/` | the FastAPI app with a real model in the MLflow format (pickle + signature, like `train.py`), loaded through the lifespan: `/health`, `/v1/predict` for T = 10, 100, 3000, 422 on bad input, degraded mode without a model, `/metrics` | nothing (untrained model built on the fly by `tests/tiny_model.py`) |
| End-to-end | `tests/e2e/` | the API **inside its Docker image** (or on Kubernetes) answers over HTTP | a running API, `E2E_BASE_URL` |

The integration and e2e tests never need MLflow, RustFS or the data: `tests/tiny_model.py` saves an
untrained `RidesClassifier` in the exact format of the registry model. Its predictions are meaningless:
the tests check the API contract, not the model quality (that is the evaluation gate's job).

`schemas.Ride` checks that each time step has exactly 6 values: 3 sensors instead of 6, or rows of
different lengths, get a 422 (`test_wrong_number_of_sensors_is_a_422`).

---

## 🔄 CI/CD

```
push / PR ──► ci.yml (GitHub runner): ruff + pytest
                 │ green on main
                 ▼
              cd.yml (self-hosted runner)
                 build image ─► e2e tests on the container ─► push Docker Hub (amd64 + arm64, tag sha-xxxxxxx)
                 ─► kubectl set image ─► rollout status ─► smoke test on prod ──(failure)──► kubectl rollout undo

manual / weekly ──► train.yml (self-hosted): dvc pull ─► dvc repro --force ─► gate
                    ─► promoted? kubectl rollout restart ─► smoke test
```

| Workflow | Trigger | Runner | Does |
|---|---|---|---|
| `ci.yml` | every push and PR | GitHub-hosted | `ruff check src tests`, `ruff format --check tests`, `pytest` |
| `cd.yml` | CI green on `main`, or manual | self-hosted | test the image, publish it, deploy it, roll back on failure |
| `train.yml` | manual (with a reason) or Monday 03:17 UTC | self-hosted | retrain on the prod MLflow, redeploy if the gate promoted the new version |

**Set up once**

0. `cd.yml` builds `docker/api.Dockerfile` and deploys to the `k8s/` manifests of part 4. The API image
   installs `prometheus-client` from `uv.lock`.
1. Self-hosted runner on the machine that runs the prod cluster: Settings → Actions → Runners → New
   self-hosted runner, then follow the commands. It needs Docker Desktop with Kubernetes on, `kubectl`
   on the `docker-desktop` context, and bash (macOS/Linux; on Windows run the runner in WSL).
2. Repository secrets (Settings → Secrets and variables → Actions):
   `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (Docker Hub access token, read/write) for `cd.yml`;
   `RUSTFS_ACCESS_KEY_ID`, `RUSTFS_SECRET_ACCESS_KEY` (DVC remote) for `train.yml`.
3. Optional variable `DOCKERHUB_IMAGE` (default `louisec13012/rides-api`).
4. Prod must already serve a model (see "Register the first prod model"): otherwise the API stays
   not ready, the rollout times out and `cd.yml` rolls back.

### Design choices (part 5)
- **CI on GitHub's runners, CD on ours.** Tests need no secret and no cluster, so they run on every
  branch and PR, isolated from our machines. Only `main` reaches the self-hosted runner, which holds
  the Docker Hub credentials and the cluster: a pull request can never run code on it.
- **The image is tested before it is published.** `cd.yml` starts the image it just built with a test
  model and runs `tests/e2e` against it; only then is it pushed and deployed.
- **One unique tag per commit** (`sha-<commit>`), never reused, as required by part 4. The deployment
  is changed with `kubectl set image`: `kubectl apply -f k8s/` would put back the tag written in the
  manifest.
- **Automatic rollback.** If the rollout or the smoke test on prod fails, `kubectl rollout undo` puts
  the previous image back. The readiness probe already keeps a pod without a model out of traffic.
- **Format check on `tests/` only.** The `src/` files are DVC dependencies: reformatting them would
  change their hash and re-run the whole pipeline. `ruff check` (lint) covers everything; `line-length`
  is set to 120 to match the code base.
- **Retraining is gated too.** `train.yml` goes through the same evaluation gate as `dvc repro`: the
  API is restarted only if the new version reached `@staging`.

## 📈 Monitoring (bonus)

```bash
docker compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up -d
```

| Service | URL |
|---|---|
| Grafana, dashboard "Rides API" | http://localhost:3000 |
| Prometheus (targets, alerts) | http://localhost:9090 |
| Raw metrics | http://localhost:8000/metrics/ |

The API exposes `/metrics/` (`src/rides/api/metrics.py`), one family per layer, as in session 6:

| Layer | Metrics | Why |
|---|---|---|
| Infrastructure | `api_requests_total{endpoint,status}`, `api_request_duration_seconds` | traffic, errors, latency |
| Inputs | `input_ride_steps`, `input_sensor_mean{sensor}` | `az` should stay near 9.81: a jump means a unit bug or a new sensor |
| Predictions | `prediction_total{label}`, `prediction_reckless_probability` | a Reckless share far above ~1 % points to drift |
| Model | `model_info{version}` | which version is served |

Alerts (`monitoring/prometheus/alerts.yml`): API down, 5xx above 5 %, p95 latency above 1 s,
Reckless share above 20 %.

**Calibrate the thresholds on a baseline run** (session 6), not on intuition: load the API, read the
p95 latency and the error rate, then set the alert thresholds above that baseline. Our baseline
(20 users, about 35 req/s on a laptop CPU, no errors): p95 about 0.4 s, hence the 1 s latency alert.

```bash
uvx locust -f monitoring/loadtest/locustfile.py --host http://localhost:8000 \
    --headless -u 20 -r 5 -t 2m --csv reports/load   # p50 / p95 / p99 in reports/load_stats.csv
```

Not covered: layer 4 (business outcome) needs the true labels of production rides, which this
project does not receive; a `POST /v1/feedback` route counting tp / fp / fn / tn would add it.

---

## 🔁 Pipeline: DVC + MLflow

```
GBG500.parquet.dvc   GBG500_labels.csv.dvc
             \           /
              prepare          → data/processed/windows.npz (raw 100-step windows, rides split train / val / test)
             /       \
         train        |        → keeps the best epoch on val, registers a new rides-classifier version
             \       /
             evaluate          → evaluation gate, sets @staging, writes reports/eval.json
```

- **Run it:** `uv run --env-file .env dvc repro`. A stage re-runs only if its code, data or `params.yaml` section changed (`uv run dvc dag` shows the graph).
- **Settings:** everything is in `params.yaml` (`prepare`: window, stride, split · `train`: hyperparameters · `evaluate`: gate). Change a value, `dvc repro`, then commit `params.yaml` + `dvc.lock` + `reports/eval.json` together.
- **Reproducibility:** every seed comes from `params.yaml` and training runs on `train.device: cpu` by default, so the same commit gives the same model on every machine (up to tiny floating-point differences between CPUs). `mps` (Apple GPU) or `cuda` is faster but not reproducible: use it for quick tries only.
- **What to commit:** `dvc.lock` and `reports/` change on every `dvc repro`, and `reports/eval.json` names a version in *your* local MLflow. Commit them **only together with a code or `params.yaml` change**; otherwise discard them (`git checkout -- dvc.lock reports/`) to avoid merge conflicts.
- **Side effects:** every run of `train` registers a new version and `evaluate` may move `@staging`, so avoid `dvc exp run` for quick tries: the experiment's model could end up in `@staging` without being committed.
- **Your MLflow is local:** `reports/train_info.json` (which run / version this pipeline registered) is cached by DVC but kept out of git, so a fresh clone always trains and registers in its own MLflow. If you wipe MLflow (`docker compose down -v`), run `uv run --env-file .env dvc repro --force train` to register again.
- **Results:** `uv run dvc metrics show` (`reports/eval.json`), `uv run dvc plots show` (confusion matrix), and in MLflow: the run's `eval/confusion_matrix.png`, the version's `eval_*` tags.
- **Evaluation gate** (`should_promote` in `src/rides/evaluate.py`): the new version gets `@staging` only if, on the test windows,
  1. its macro F1 beats the "always Safe" baseline by at least `evaluate.min_gain` (0.05),
  2. its Reckless recall is above the baseline's (0),
  3. at most `evaluate.max_false_alarm_rate` (10 %) of the *Safe overall*, *Bad weather* and *Tandem* windows are flagged as *Reckless*,
  4. its macro F1 is not below the model currently in `@staging` (re-scored on the same test set).

  A failed gate does not fail `dvc repro`: it is recorded in `reports/eval.json` (`"promoted": false`) and in the version's `eval_gate` / `eval_reason` tags.
- **Model format:** logged as pickle with an explicit signature `(-1, -1, 6)`, so any batch size and any T ≥ 10 work through `mlflow.pyfunc` (MLflow's default pt2 format would freeze batch = 1 and T = 100). The API must send `float32`. The modules the model needs (`rides/data.py`, `rides/model.py`) are bundled with it (`code_paths`), so it loads even where the package is not installed; this relies on MLflow's defaults `MLFLOW_ALLOW_PICKLE_DESERIALIZATION=true` and `MLFLOW_ALLOW_MODEL_CODE_EXECUTION=true`.
- **Known limit, Reckless:** the dataset has only 6 *Reckless* rides, so the test set holds **1** (about 60 windows) and the validation set 1. The Reckless recall used by the gate therefore depends on a single ride: treat it as a smoke test, not as a reliable measure.
- **Known limit:** if you re-run `evaluate` on the version already in `@staging` with a stricter gate, it is tagged `failed` but keeps `@staging` (the API is never left without a model).

### Data with DVC
The remote is the `dvc` bucket on **your own** local RustFS (`localhost:9000`), so the first time it is empty. Seed it once: copy `GBG500.parquet` and `GBG500_labels.csv` into `data/raw/`, run `uv run dvc add data/raw/GBG500.parquet data/raw/GBG500_labels.csv` (`git diff` must show no change to the `.dvc` files: same data, same hash), then `uv run dvc push`. After that, use `uv run dvc pull data/raw` and `uv run dvc push`. Do not run a bare `dvc pull`: `dvc.lock` also lists pipeline outputs (`windows.npz`, `train_info.json`, ...) that only exist in the cache of the machine that produced them, so it would report missing files. `dvc repro` rebuilds them instead.

Windows: `PYTHONUTF8=1` in `.env` (already in `.env.example`) avoids a crash when MLflow prints emojis under `dvc repro`. Line endings are forced to LF (`.gitattributes`) because DVC hashes the bytes of the code it depends on: CRLF on one machine would make every stage look changed on the others.

---

## 📁 Repository structure

```
├── src/rides/            # data, model, training, evaluation, API
├── webapp/               # Streamlit demo
├── tests/                # unit / integration / e2e
├── docker/               # Dockerfiles
├── k8s/                  # Kubernetes manifests (prod)
├── .github/workflows/    # CI / CD / retraining
├── monitoring/           # Prometheus + Grafana (bonus)
├── docker-compose.yml    # dev environment
├── dvc.yaml, params.yaml # pipeline DAG
└── docs/screenshots/
```

