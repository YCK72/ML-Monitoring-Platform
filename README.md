# ML Monitoring Platform

A production-grade ML model monitoring platform that detects data drift in real time, fires Slack alerts when drift exceeds configurable thresholds, and visualises model health through a multi-page React dashboard.

Built over a structured 14-day project across the full MLOps stack: from model training and Kafka-based prediction ingestion, through statistical drift detection, to a live alerting pipeline and interactive frontend.

---

## Demo

> Dashboard screenshot — Overview page showing per-feature drift severity

![Overview Dashboard]![img.png](img.png)

---

## What It Does

1. **Serves predictions** via a FastAPI endpoint backed by an XGBoost model registered in MLflow
2. **Streams prediction events** into Kafka in real time
3. **Consumes and persists** events into PostgreSQL via a rolling evaluation window
4. **Detects drift** every 2 minutes using KS Test, PSI, Chi-Squared, and Wasserstein Distance across 15 features
5. **Classifies severity** as Green / Yellow / Red per feature and overall
6. **Fires alerts** to Slack (and optionally email) with cooldown deduplication and escalation bypass
7. **Exposes a REST API** for the frontend and downstream consumers
8. **Visualises everything** in a multi-page React dashboard with live data

---

## Tech Stack

### Backend
| Layer | Technology |
|---|---|
| Model training | XGBoost, scikit-learn |
| Model registry | MLflow |
| Prediction API | FastAPI |
| Event streaming | Apache Kafka |
| Database | PostgreSQL + SQLAlchemy + Alembic |
| Drift detection | KS Test, PSI, Chi-Squared, Wasserstein |
| Alerting | Custom engine — Slack webhooks, SMTP email |
| Scheduling | APScheduler |
| HTML reports | Evidently AI |
| Containerisation | Docker Compose |

### Frontend
| Layer | Technology |
|---|---|
| Framework | React 18 + TypeScript |
| Build tool | Vite |
| Routing | React Router v6 |
| Data fetching | TanStack Query (React Query) |
| HTTP client | Axios |
| Styling | Tailwind CSS v4 |
| Component library | shadcn/ui (Radix primitives) |
| Charts | Recharts |

---

## Architecture

```
                    MLflow Model Registry
                           │
                           ▼
              Prediction Service (FastAPI :8001)
                           │
                      Kafka Topic
                    prediction-events
                           │
                           ▼
                    Kafka Consumer
                           │
                           ▼
                      PostgreSQL
                  ┌────────────────┐
                  │ prediction_    │
                  │ records        │
                  └────────────────┘
                           │
                           ▼
            Scheduled Drift Evaluator (2 min)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      Drift Detection              Evidently
      Engine                       HTML Reports
      KS / PSI / Chi² /
      Wasserstein
              │
              ▼
       Severity Engine
       Green / Yellow / Red
              │
       ┌──────┴──────┐
       ▼             ▼
  drift_reports  Alert Engine
  (Postgres)     + Cooldown Logic
                      │
               ┌──────┴──────┐
               ▼             ▼
            Slack          Email
           Webhook          SMTP
                      │
                      ▼
             FastAPI REST API (:8000)
                      │
                      ▼
             React Dashboard (:5173)
        ┌─────────────┬──────────────┐
        ▼             ▼              ▼
    Overview    Drift Timeline    Alerts
    Feature     Models
    Detail
```

---

## Project Structure

```
ML-Monitoring-Platform/
├── backend/
│   ├── config/
│   │   └── alert_thresholds.yaml   # Severity thresholds and cooldown config
│   ├── scripts/
│   │   ├── generate_batch.py       # Generate normal/mild/heavy drift test data
│   │   └── send_predictions.py     # POST batches to prediction service
│   ├── src/
│   │   ├── alerting/               # Alert engine, dispatcher, Slack/email notifiers
│   │   ├── api/                    # FastAPI app and routers
│   │   ├── drift/                  # Drift detection, evaluator, scheduler, Evidently
│   │   ├── monitoring/             # SQLAlchemy models, database, repository
│   │   ├── prediction/             # Prediction service, Kafka producer
│   │   └── training/               # Model training, data generation, MLflow
│   └── tests/
│       └── unit/
├── frontend/
│   └── src/
│       ├── api/                    # Axios client + React Query hooks
│       ├── components/
│       │   ├── layout/             # Persistent sidebar navigation
│       │   └── ui/                 # shadcn/ui components
│       ├── pages/                  # Overview, Alerts, DriftTimeline, FeatureDetail, Models
│       └── types/                  # TypeScript interfaces matching backend schemas
├── docker-compose.yml
└── Makefile
```

---

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Node.js 18+
- npm 9+

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YCK72/ML-Monitoring-Platform.git
cd ML-Monitoring-Platform
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your values:
```
POSTGRES_USER=mluser
POSTGRES_PASSWORD=mlpassword
POSTGRES_DB=ml_monitoring
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
```

Create `backend/.env`:
```
API_KEY=your_random_api_key_here
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
DATABASE_URL=postgresql+psycopg2://mluser:mlpassword@localhost:5433/ml_monitoring
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
MLFLOW_TRACKING_URI=http://localhost:5000
DRIFT_EVAL_INTERVAL_MINUTES=2
```

Create `frontend/.env`:
```
VITE_API_KEY=your_random_api_key_here
```

### 3. Start infrastructure

```bash
docker compose up -d
```

This starts: PostgreSQL (port 5433), Kafka, Zookeeper, and MLflow (port 5000).

### 4. Set up the Python environment

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 5. Run database migrations

```bash
alembic upgrade head
```

### 6. Train and register the model

```bash
python -m src.training.train
python -m src.training.sync_model_to_postgres
```

### 7. Start backend services (three separate terminals)

**Terminal 1 — REST API:**
```bash
cd backend
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 — Prediction Service:**
```bash
cd backend
uvicorn src.prediction.service:app --reload --port 8001
```

**Terminal 3 — Drift Scheduler:**
```bash
cd backend
python -m src.drift.scheduler
```

### 8. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

### 9. Generate and send drift data

```bash
cd backend
python scripts/generate_batch.py --mode heavy --n 1000 --out heavy_batch.json
python scripts/send_predictions.py --file heavy_batch.json --url http://localhost:8001/predict
```

Wait ~2 minutes for the scheduler to evaluate. The dashboard will update and Slack alerts will fire.

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | Per-feature severity cards with KS p-value and PSI score |
| **Drift Timeline** | Line chart of overall severity across historical reports |
| **Feature Detail** | Select any feature to see its historical PSI trend |
| **Models** | Registered model versions, stages, and training metrics |
| **Alerts** | Paginated alert history with severity, channel, and timestamp |

---

## REST API

All endpoints require `X-API-Key` header.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/drift/summary` | Latest drift report summary |
| GET | `/drift/reports` | Paginated drift report history |
| GET | `/drift/reports/{id}` | Single drift report |
| GET | `/alerts` | Paginated alert history |
| GET | `/models` | Registered model versions |
| GET | `/metrics/history` | Historical metrics |

---

## Alert Configuration

Edit `backend/config/alert_thresholds.yaml`:

```yaml
yellow:
  channels: [slack]
  cooldown_minutes: 30

red:
  channels: [slack, email]
  cooldown_minutes: 30

allow_escalation_during_cooldown: true
```

- **Cooldown**: suppresses repeat alerts for the same feature within the window
- **Escalation bypass**: a severity increase (Yellow → Red) always fires, even inside the cooldown

---

## Known Gaps

- **Prediction drift**: feature drift is fully operational; Wasserstein-based prediction drift requires baseline probability scoring of the reference dataset (not yet implemented)
- **CI/CD**: no GitHub Actions workflows yet
- **Test coverage**: unit tests exist for the repository layer; broader coverage not yet implemented
- **Dark mode**: not implemented in the current dashboard iteration

---

## Development Notes

- The `.env` files are excluded from version control — never commit real credentials
- Kafka consumer uses a time-windowed query; predictions sent in a previous session won't be picked up after a scheduler restart
- `DRIFT_EVAL_INTERVAL_MINUTES=2` in `.env` speeds up iteration during development
- The prediction service (port 8001) and REST API (port 8000) are separate processes

---

## License

MIT