# Eduverse Backend – FastAPI

Python FastAPI backend for the Eduverse / Genverse.ai EdTech platform.
Replaces the Supabase backend with a self-hosted stack: **FastAPI + PostgreSQL + SQLAlchemy + Alembic**.

---

## Prerequisites

| Tool | Minimum version |
|---|---|
| Python | 3.11+ |
| PostgreSQL | 14+ |
| pip | latest |
| (Optional) Redis | 6+ — for future caching/rate-limiting |

---

## 1. Clone / Navigate to the Backend Directory

```bash
cd "FGIL Projects/Eduverse/eduverse-backend"
```

---

## 2. Create & Activate a Virtual Environment

```bash
# Create
python -m venv venv

# Activate – Linux / macOS
source venv/bin/activate

# Activate – Windows (PowerShell)
venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in every value:

```env
# --- Database (connection URLs are built automatically from these values) ---
DB_HOST=localhost
DB_PORT=5432
DB_NAME=eduverse_db
DB_USER=postgres
DB_PASSWORD=your_password

# --- JWT ---
SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=30

# --- AI Providers ---
GOOGLE_GEMINI_API_KEY=<your-gemini-api-key>
OPENAI_API_KEY=<your-openai-api-key>       # optional fallback

# --- Storage ---
STORAGE_ROOT=./uploads
MAX_UPLOAD_SIZE_MB=50

# --- CORS (comma-separated origins of your React frontend) ---
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

---

## 5. Create the PostgreSQL Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Inside psql
CREATE DATABASE eduverse;
\q
```

---

## 6. Generate & Run Database Migrations

Alembic is configured for **autogenerate** — it reads your SQLAlchemy models and produces
migration scripts automatically. You never write migration SQL by hand.

### First-time setup (empty database)

```bash
# 1. Auto-generate the initial migration from your models
alembic revision --autogenerate -m "initial schema"

# 2. Apply it to the database
alembic upgrade head
```

Alembic creates a timestamped file inside `alembic/versions/`. Review it before applying
to confirm it matches your models, then run `upgrade head`.

### Every time you change a model

```bash
# 1. Edit your model file in app/models/
# 2. Generate a migration for the change
alembic revision --autogenerate -m "add column X to table Y"

# 3. Apply
alembic upgrade head
```

### Other useful commands

```bash
alembic current          # show which revision the DB is on
alembic history          # list all migration revisions
alembic downgrade -1     # roll back one step
alembic downgrade base   # roll back everything (drops all tables)
```

---

## 7. Start the Development Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now running at **http://localhost:8000**

---

## 8. Explore the API

| URL | Description |
|---|---|
| http://localhost:8000/docs | Swagger UI – interactive API docs |
| http://localhost:8000/redoc | ReDoc – alternate documentation |
| http://localhost:8000/health | Health check endpoint |
| http://localhost:8000/api/v1/... | All API routes |

---

## 9. Connect the Frontend

In your React frontend project, set the base API URL to point at the running backend:

```env
# frontend/.env  (or .env.local)
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

Update all Supabase client calls to use this base URL instead.
CORS is pre-configured for `localhost:5173` and `localhost:3000` (adjust `CORS_ORIGINS` in `.env` for other ports or production domains).

---

## 10. Production Deployment (Quick Reference)

```bash
# Run with multiple workers using Gunicorn
pip install gunicorn
gunicorn app.main:app -k uvicorn.workers.UvicornWorker \
  --workers 4 --bind 0.0.0.0:8000

# Or with uvicorn directly (no auto-reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Set `STORAGE_ROOT` to an absolute path (e.g. `/var/www/eduverse/uploads`) and configure a reverse proxy (Nginx / Caddy) to serve static files from that path for performance.

---

## Project Structure

```
eduverse-backend/
├── alembic/                  # Alembic migration environment
│   ├── versions/              # auto-generated migration files go here
│   ├── env.py
│   └── script.py.mako
├── app/
│   ├── core/
│   │   ├── exceptions.py     # Custom HTTP exceptions
│   │   └── security.py       # JWT + password hashing
│   ├── models/               # SQLAlchemy ORM models (11 files)
│   ├── routers/              # FastAPI route handlers (25 files)
│   ├── schemas/              # Pydantic request/response schemas (13 files)
│   ├── services/             # Business logic layer
│   │   ├── ai_service.py     # Gemini / OpenAI wrapper
│   │   ├── points_service.py # Atomic point deduction
│   │   └── storage_service.py# Local filesystem storage
│   ├── config.py             # Pydantic Settings (loads .env)
│   ├── database.py           # Async SQLAlchemy engine + session
│   ├── dependencies.py       # FastAPI dependency injection
│   └── main.py               # App entry point, middleware, router registration
├── uploads/                  # File upload storage (auto-created)
├── .env.example              # Environment variable template
├── .gitignore
├── alembic.ini
└── requirements.txt
```

---

## Key Environment Variable Reference

| Variable | Description |
|---|---|
| `DB_HOST` | PostgreSQL host (default `localhost`) |
| `DB_PORT` | PostgreSQL port (default `5432`) |
| `DB_NAME` | Database name |
| `DB_USER` | Database username |
| `DB_PASSWORD` | Database password |
| `SECRET_KEY` | Random hex string for JWT signing |
| `GOOGLE_GEMINI_API_KEY` | Primary AI provider key |
| `OPENAI_API_KEY` | Fallback AI provider key (optional) |
| `STORAGE_ROOT` | Directory where uploaded files are saved |
| `MAX_UPLOAD_SIZE_MB` | Max allowed file upload size |
| `CORS_ORIGINS` | Comma-separated list of allowed frontend origins |
