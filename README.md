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

### How we measure success
The model only needs to **beat chance**: infrastructure is what is graded. But with 83 % *Safe*, a model that always answers "Safe" gets 83 % accuracy while detecting nothing. We therefore track:
- **macro F1** across the 5 classes;
- **recall on *Reckless***;
- **false alarm rate** on *Safe Overall*, *Bad Weather* and *Tandem*.

**Baseline to beat:** always predict the majority class.
Because *Reckless* has only 6 rides, we use **stratified cross-validation** and split each ride into **fixed-size windows** to get more training samples.

---

##  Architecture

```
data (Minio + DVC) → training (PyTorch) → tracking & registry (MLflow, artifacts on Minio)
      → API (FastAPI) → Docker image (Docker Hub) → Kubernetes ← CI/CD GitHub Actions (self-hosted runner)
                                       ↑
                                 demo web app
```

| Environment | Where | Services |
|---|---|---|
| **dev** | local machine, `docker compose` | Minio, MLflow, Postgres, API, web app |
| **prod** | local Kubernetes (**Docker Desktop**) | **all services**: Minio, MLflow, Postgres, API, web app (Helm charts where available) |

---

## ✅ Requirements coverage

| # | Requirement | Where |
|---|---|---|
| 1 | Extract data | `src/rides/data.py` |
| 2 | Deep learning classification model (5 classes) | `src/rides/model.py`, `src/rides/train.py` (PyTorch) |
| 3 | Model stored on S3 (Minio) | MLflow artifacts → bucket `mlflow` |
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
- [ ] Explore `GBG500.parquet`: IMU columns, sampling rate, ride lengths, class imbalance
- [ ] `data.py`: loading, join with labels, fixed-size windowing, normalisation, stratified split
- [ ] `model.py`: small PyTorch network (1D CNN or LSTM)
- [ ] `train.py`: training + `mlflow.log_params / log_metrics` + register the model as **`rides-classifier`**
- [ ] Handle class imbalance (class weights or oversampling)
- **Hands over:** the exact model input format: window length **T**, list of the **C** sensor columns, normalisation stats

### 2 · MLflow + pipeline — @handle2
- [ ] `evaluate.py`: macro F1, Reckless recall, confusion matrix logged to MLflow; promote to **`@staging`** if better than baseline
- [ ] DVC: `dvc init`, Minio remote (bucket `dvc`), `dvc add data/raw/GBG500.parquet`, `dvc push`
- [ ] `dvc.yaml`: `prepare → train → evaluate` DAG (the required pipeline DAG)
- [ ] `params.yaml`: hyperparameters
- **Hands over:** a `models:/rides-classifier@staging` model in MLflow

### 3 · API + web app — @handle3
- [ ] `src/rides/api/`: `/health`, `POST /v1/predict`, Swagger docs (`description`, `examples`)
- [ ] Adapt `schemas.py` to the input format from @Andrianrjs
- [ ] `webapp/app.py` (Streamlit): pick or upload a ride, call the API, show the class and probabilities
- 💡 Until the real model exists: test with a dummy model returning random probabilities
- **Hands over:** an API running with `uv run uvicorn rides.api.main:app`

### 4 · Docker + Kubernetes — @handle4
- [ ] `docker/api.Dockerfile`, `docker/webapp.Dockerfile`
- [ ] Add the API and web app to `docker-compose.yml` (**dev**)
- [ ] Enable Kubernetes in **Docker Desktop** (Settings → Kubernetes)
- [ ] `k8s/`: Deployment + Service + readiness/liveness probes on `/health` for the API and web app
- [ ] Deploy **MLflow, Minio and Postgres on the cluster too** (prod = all services), using existing Helm charts
- [ ] Push images to **Docker Hub** (URLs go in the submission email)
- **Hands over:** `docker compose up` starts dev, `kubectl apply -f k8s/` (+ Helm) starts prod

### 5 · CI/CD + tests + bonus — @handle5
- [ ] Set up a **self-hosted GitHub runner** (Settings → Actions → Runners)
- [ ] `ci.yml`: lint + `pytest` on every push and PR
- [ ] `cd.yml`: build → push to Docker Hub → deploy to Kubernetes (`runs-on: self-hosted`)
- [ ] `tests/unit`, `tests/integration` (API with `TestClient`), `tests/e2e` (API in Docker → request → valid response)
- [ ] Bonus: `train.yml` retraining workflow (pull data → retrain → redeploy)
- [ ] Bonus: Prometheus + Grafana (API latency, request count, prediction distribution)
- [ ] Bonus: feature store (e.g. Feast) for the windowed IMU features

---

##  Contracts between parts
Fixed now so everyone can work in parallel. **Do not change without telling the team.**

| Item | Value |
|---|---|
| MLflow model name | `rides-classifier` |
| Alias used by the API | `@staging` |
| Model input | `float32`, shape `(batch, T, C)` *(T and C set by @Andrianrjs)* |
| Model output | probabilities, shape `(batch, 5)`, order: Safe, Safe Overall, Bad Weather, Tandem, Reckless |
| API route | `POST /v1/predict` |
| Minio buckets | `mlflow` (artifacts), `dvc` (data) |


---

## 🛠️ Getting started

```bash
git clone https://github.com/Andrianrjs/mlops-rides.git && cd mlops-rides
uv sync
cp .env.example .env
docker compose up -d     # MLflow → http://localhost:5000 · Minio console → http://localhost:9001
# Data: copy GBG500.parquet and GBG500_labels.csv into data/raw/ (until `dvc pull` is set up)
uv run uvicorn rides.api.main:app --reload   # Swagger → http://localhost:8000/docs
```

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

