<!--
How to use this file once fix/review-parts-1-2 is merged (README.md is edited on that branch):
  1. Replace the "### 4 · Docker + Kubernetes" checklist with section A.
  2. Paste sections B, C and D right after "## 🛠️ Getting started" (before "## 🔁 Pipeline").
  3. Replace "Minio" with "RustFS" in the README (lines about the architecture, environments,
     requirements, contracts and getting started). In "Getting started", the storage console
     is now "RustFS console → http://localhost:9001".
  4. Lines marked (after PR #4) are for the web app: fill them in once it is merged.
  5. Delete this comment.
-->

<!-- ===== A. replaces the part 4 checklist ===== -->

### 4 · Docker + Kubernetes — @lcaquelin
- [x] Clean up the Docker setup: pinned image versions, Postgres and MLflow healthchecks, MinIO services renamed to RustFS, outdated `docker/mlflow/docker-compose.yml` removed
- [x] `docker/api.Dockerfile` (CPU-only torch, non-root user, healthcheck) and `.dockerignore`
- [ ] `docker/webapp.Dockerfile` (after PR #4)
- [x] API added to `docker-compose.yml` (dev) · [ ] web app (after PR #4)
- [x] Kubernetes enabled in Docker Desktop (kubeadm, 1 node)
- [x] `k8s/`: RustFS, Postgres, MLflow and API on the cluster (prod = all services), with readiness / liveness probes and resource limits · [ ] web app (after PR #4)
- [x] Images pushed to Docker Hub
- **Hands over:** `docker compose up -d` starts dev · `kubectl apply -f k8s/` starts prod

<!-- ===== B. new section ===== -->

## 🐳 Docker images

| Image | Docker Hub | Built from |
|---|---|---|
| API | [louisec13012/rides-api](https://hub.docker.com/r/louisec13012/rides-api) | `docker/api.Dockerfile` |
| MLflow server | [louisec13012/rides-mlflow](https://hub.docker.com/r/louisec13012/rides-mlflow) | `docker/mlflow/Dockerfile` |
| Web app | (after PR #4) | `docker/webapp.Dockerfile` |

`docker compose build` builds them under these names, so dev and prod run the same images.
To publish a new version: change the tag (`0.1.0`) in `docker-compose.yml` and `k8s/`, then
`docker compose build` and `docker push louisec13012/rides-api:<tag>` (same for `rides-mlflow`).

<!-- ===== C. new section ===== -->

## 🧪 Run in dev (docker compose)

Every service runs in Docker on your machine; you run training from your terminal.
Prerequisite: the data in `data/raw/` (see Getting started).

```bash
cp .env.example .env                          # Windows: Copy-Item .env.example .env
docker compose up -d --build --remove-orphans # RustFS, Postgres, MLflow, API
uv run --env-file .env dvc repro              # trains and puts a model in @staging
docker compose restart api                    # the API loads the @staging model at startup
curl http://localhost:8000/health             # Windows: curl.exe → {"status":"ok", ...}
```

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5001 |
| API (Swagger) | http://localhost:8000/docs |
| RustFS console | http://localhost:9001 |
| Web app | (after PR #4) |

`--remove-orphans` stops the old `minio` / `minio-init` containers if you started the stack
before they were renamed; your data is kept (the volume is still named `minio-data`).

<!-- ===== D. new section ===== -->

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
kubectl create secret generic rides-secrets -n rides --from-env-file=.env  # credentials never go to git
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
| Web app (prod) | (after PR #4) |

**Remove everything:** `kubectl delete namespace rides` (this also deletes the prod data).

### Design choices (part 4)
- **CPU-only torch in the API image.** On Linux, the default torch build pulls several GB of NVIDIA CUDA libraries that are useless without a GPU. The Dockerfile installs every dependency pinned by `uv.lock` except torch, then torch at the same version from the PyTorch CPU index: about 490 MB to download instead of several GB.
- **Readiness vs liveness.** The API's `/health` answers 503 until a model is loaded. It drives the **readiness** probe (no traffic before a model is there); the **liveness** probe only checks the port answers, so a missing model does not cause a restart loop.
- **MLflow serves the model files** (`--serve-artifacts`): the API needs the MLflow URL only, no storage credentials.
- **MLflow host check.** Since MLflow 3.5, the server rejects unknown Host headers (DNS rebinding protection). `MLFLOW_SERVER_ALLOWED_HOSTS` allows `localhost:*` and `mlflow:5000` (the API inside Docker or Kubernetes); the Kubernetes probes call `localhost` for the same reason.
- **2 MLflow workers.** The default (4) went over the 3 GB memory limit on the cluster (OOMKilled).
- **Secrets** come from `.env` (dev) or a Kubernetes Secret created from it (prod); no credential is committed or baked into an image.
- **Plain manifests rather than Helm:** the same images and settings as in dev, one file per service, easy to read and explain.
