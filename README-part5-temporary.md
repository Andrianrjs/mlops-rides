<!--
How to use this file (same approach as README-part4-temporary.md):
1. Replace the "### 5 · CI/CD + tests + bonus" checklist with section A.
2. Paste sections B and C after the "Deploy to prod" section (part 4).
3. Delete this comment.
-->

<!-- ===== A. replaces the part 5 checklist ===== -->

### 5 · CI/CD + tests + bonus — @clemencemarco-prog
- [x] `ci.yml`: ruff lint + unit and integration tests on every push and PR (GitHub-hosted runner)
- [x] `cd.yml`: build → e2e test of the image → push to Docker Hub (multi-arch) → deploy to Kubernetes → smoke test, automatic rollback (`runs-on: self-hosted`)
- [x] `tests/unit`, `tests/integration` (API + real MLflow model with TestClient), `tests/e2e` (API in Docker → request → valid response)
- [x] Bonus: `train.yml` retraining workflow (pull data → retrain → gate → redeploy if promoted)
- [x] Bonus: Prometheus + Grafana (API latency, request count, input and prediction distribution, alerts)
- [ ] Self-hosted runner registered (Settings → Actions → Runners) on the machine running the prod cluster
- [ ] Bonus: feature store (Feast), not planned
- **Hands over:** a push to `main` that passes CI is tested, published and deployed without manual steps

<!-- ===== B. new section ===== -->

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

**Known issue, tracked by a test:** `schemas.Ride` does not check that each time step has 6 values, so
3 sensors instead of 6, or rows of different lengths, crash the model and return a 500 instead of a 422.
`test_wrong_number_of_sensors_is_a_422` is marked `xfail(strict=True)`: once the schema is fixed, the
test passes and pytest asks to remove the mark.

<!-- ===== C. new section ===== -->

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

0. `cd.yml` builds `docker/api.Dockerfile` and deploys to the `k8s/` manifests of part 4: merge
   `feat/docker-k8s` before this branch. The API image installs `prometheus-client` from `uv.lock`.
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
