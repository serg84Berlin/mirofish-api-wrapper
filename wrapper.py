"""MiroFish Offline API Wrapper – Flask Proxy Server.

A REST server on port 5050 that orchestrates the full MiroFish simulation
pipeline (6 stages) in background threads and exposes a simple API for the
frontend.
"""

import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

# ======================================================================
# MiroFish low-level client
# ======================================================================

MIROFISH_BASE_URL = os.getenv("MIROFISH_BASE_URL", "http://localhost:8000")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "1200"))


class MiroFishAPIError(Exception):
    def __init__(self, status_code: int, detail: Any = None):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class MiroFishClient:
    def __init__(self, base_url: str = MIROFISH_BASE_URL, timeout: int = LLM_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.request(method, self._url(path), **kwargs)
        if not resp.ok:
            raise MiroFishAPIError(resp.status_code, resp.text)
        ct = resp.headers.get("content-type", "")
        if "application/json" in ct:
            return resp.json()
        return resp.text

    # Health
    def health(self) -> Any:
        return self._request("GET", "/health")

    # Graph / Ontology
    def generate_ontology(self, payload: dict) -> Any:
        return self._request("POST", "/api/graph/ontology/generate", json=payload)

    def build_graph(self, payload: dict) -> Any:
        return self._request("POST", "/api/graph/build", json=payload)

    # Simulation
    def create_simulation(self, payload: dict) -> Any:
        return self._request("POST", "/api/simulation/create", json=payload)

    def generate_profiles(self, payload: dict) -> Any:
        return self._request("POST", "/api/simulation/generate-profiles", json=payload)

    def start_simulation(self, payload: dict) -> Any:
        return self._request("POST", "/api/simulation/start", json=payload)

    def get_run_status(self, simulation_id: str) -> Any:
        return self._request("GET", f"/api/simulation/{simulation_id}/run-status")

    def wait_for_simulation(
        self, simulation_id: str, poll_interval: float = 5.0, timeout: Optional[float] = None
    ) -> Any:
        elapsed = 0.0
        while True:
            status = self.get_run_status(simulation_id)
            state = status if isinstance(status, str) else status.get("status", "")
            if state in ("completed", "failed", "error"):
                return status
            if timeout is not None and elapsed >= timeout:
                raise TimeoutError(
                    f"Simulation {simulation_id} did not finish within {timeout}s"
                )
            time.sleep(poll_interval)
            elapsed += poll_interval

    # Report
    def generate_report(self, payload: dict) -> Any:
        return self._request("POST", "/api/report/generate", json=payload)

    def report_chat(self, payload: dict) -> Any:
        return self._request("POST", "/api/report/chat", json=payload)


# ======================================================================
# 27 EU country profiles (Eurostat-based reference data)
# ======================================================================

EU_COUNTRIES: list[dict] = [
    {"code": "AT", "name": "Austria",        "population": 9104772,  "gdp_per_capita": 53637, "unemployment_rate": 5.0, "capital": "Vienna"},
    {"code": "BE", "name": "Belgium",        "population": 11686140, "gdp_per_capita": 49540, "unemployment_rate": 5.5, "capital": "Brussels"},
    {"code": "BG", "name": "Bulgaria",       "population": 6447710,  "gdp_per_capita": 13980, "unemployment_rate": 4.2, "capital": "Sofia"},
    {"code": "HR", "name": "Croatia",        "population": 3862305,  "gdp_per_capita": 18570, "unemployment_rate": 6.1, "capital": "Zagreb"},
    {"code": "CY", "name": "Cyprus",         "population": 920701,   "gdp_per_capita": 31450, "unemployment_rate": 6.0, "capital": "Nicosia"},
    {"code": "CZ", "name": "Czech Republic", "population": 10827529, "gdp_per_capita": 27870, "unemployment_rate": 2.6, "capital": "Prague"},
    {"code": "DK", "name": "Denmark",        "population": 5932654,  "gdp_per_capita": 67790, "unemployment_rate": 4.8, "capital": "Copenhagen"},
    {"code": "EE", "name": "Estonia",        "population": 1365884,  "gdp_per_capita": 28350, "unemployment_rate": 6.4, "capital": "Tallinn"},
    {"code": "FI", "name": "Finland",        "population": 5563970,  "gdp_per_capita": 53650, "unemployment_rate": 7.2, "capital": "Helsinki"},
    {"code": "FR", "name": "France",         "population": 68042591, "gdp_per_capita": 44408, "unemployment_rate": 7.3, "capital": "Paris"},
    {"code": "DE", "name": "Germany",        "population": 84482267, "gdp_per_capita": 51383, "unemployment_rate": 3.0, "capital": "Berlin"},
    {"code": "GR", "name": "Greece",         "population": 10394055, "gdp_per_capita": 20867, "unemployment_rate": 11.0, "capital": "Athens"},
    {"code": "HU", "name": "Hungary",        "population": 9597085,  "gdp_per_capita": 18728, "unemployment_rate": 4.1, "capital": "Budapest"},
    {"code": "IE", "name": "Ireland",        "population": 5194336,  "gdp_per_capita": 103685, "unemployment_rate": 4.3, "capital": "Dublin"},
    {"code": "IT", "name": "Italy",          "population": 58850717, "gdp_per_capita": 35657, "unemployment_rate": 7.6, "capital": "Rome"},
    {"code": "LV", "name": "Latvia",         "population": 1883008,  "gdp_per_capita": 21148, "unemployment_rate": 6.8, "capital": "Riga"},
    {"code": "LT", "name": "Lithuania",      "population": 2857279,  "gdp_per_capita": 24030, "unemployment_rate": 6.9, "capital": "Vilnius"},
    {"code": "LU", "name": "Luxembourg",     "population": 660809,   "gdp_per_capita": 126426, "unemployment_rate": 4.9, "capital": "Luxembourg"},
    {"code": "MT", "name": "Malta",          "population": 542051,   "gdp_per_capita": 33486, "unemployment_rate": 3.0, "capital": "Valletta"},
    {"code": "NL", "name": "Netherlands",    "population": 17811291, "gdp_per_capita": 57768, "unemployment_rate": 3.5, "capital": "Amsterdam"},
    {"code": "PL", "name": "Poland",         "population": 36753736, "gdp_per_capita": 18321, "unemployment_rate": 2.8, "capital": "Warsaw"},
    {"code": "PT", "name": "Portugal",       "population": 10467366, "gdp_per_capita": 25065, "unemployment_rate": 6.5, "capital": "Lisbon"},
    {"code": "RO", "name": "Romania",        "population": 19038098, "gdp_per_capita": 15792, "unemployment_rate": 5.4, "capital": "Bucharest"},
    {"code": "SK", "name": "Slovakia",       "population": 5428792,  "gdp_per_capita": 21088, "unemployment_rate": 5.8, "capital": "Bratislava"},
    {"code": "SI", "name": "Slovenia",       "population": 2116972,  "gdp_per_capita": 29291, "unemployment_rate": 3.7, "capital": "Ljubljana"},
    {"code": "ES", "name": "Spain",          "population": 48059777, "gdp_per_capita": 30996, "unemployment_rate": 11.7, "capital": "Madrid"},
    {"code": "SE", "name": "Sweden",         "population": 10551707, "gdp_per_capita": 55873, "unemployment_rate": 7.5, "capital": "Stockholm"},
]

_COUNTRY_BY_CODE: dict[str, dict] = {c["code"]: c for c in EU_COUNTRIES}


def build_seed_document(country_code: str, scenario: str = "") -> str:
    """Generate a seed document for a country that feeds MiroFish ontology generation."""
    country = _COUNTRY_BY_CODE.get(country_code.upper())
    if country is None:
        raise ValueError(f"Unknown country code: {country_code}")

    lines = [
        f"Country: {country['name']} ({country['code']})",
        f"Capital: {country['capital']}",
        f"Population: {country['population']:,}",
        f"GDP per capita (EUR): {country['gdp_per_capita']:,}",
        f"Unemployment rate: {country['unemployment_rate']}%",
        "",
        "Context: European Union member state.",
        f"Scenario focus: {scenario}" if scenario else "",
        "",
        "The simulation should model socio-economic dynamics including labour markets, "
        "fiscal policy, demographic shifts, trade flows, and public infrastructure "
        f"investment for {country['name']}. Use Eurostat baseline indicators where "
        "applicable.",
    ]
    return "\n".join(line for line in lines if line is not None)


# ======================================================================
# In-memory job store
# ======================================================================

class JobStore:
    """Thread-safe in-memory store for simulation jobs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def create(self, country_code: str, scenario: str, params: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "country_code": country_code.upper(),
            "scenario": scenario,
            "params": params,
            "status": "queued",
            "stage": None,
            "stages_completed": [],
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[dict]:
        with self._lock:
            return list(self._jobs.values())

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(fields)
            job["updated_at"] = datetime.now(timezone.utc).isoformat()


store = JobStore()
mf = MiroFishClient()

# ======================================================================
# Pipeline orchestration (runs in background thread)
# ======================================================================

PIPELINE_STAGES = [
    "ontology_generate",
    "graph_build",
    "simulation_create",
    "generate_profiles",
    "simulation_start",
    "simulation_run",
    "report_generate",
]


def _run_pipeline(job_id: str) -> None:
    """Execute all MiroFish stages sequentially for a job."""
    job = store.get(job_id)
    if job is None:
        return

    country_code = job["country_code"]
    scenario = job["scenario"]
    params = job["params"]
    seed_doc = build_seed_document(country_code, scenario)

    store.update(job_id, status="running", stage="ontology_generate", progress=0)

    try:
        # Stage 1 – Ontology generation
        ontology_payload = {"seed_document": seed_doc, **params.get("ontology", {})}
        ontology_result = mf.generate_ontology(ontology_payload)
        store.update(
            job_id,
            stage="graph_build",
            stages_completed=["ontology_generate"],
            progress=14,
        )

        # Stage 2 – Graph build
        graph_payload = {"ontology": ontology_result, **params.get("graph", {})}
        graph_result = mf.build_graph(graph_payload)
        store.update(
            job_id,
            stage="simulation_create",
            stages_completed=["ontology_generate", "graph_build"],
            progress=28,
        )

        # Stage 3 – Create simulation
        sim_create_payload = {"graph": graph_result, **params.get("simulation", {})}
        sim_result = mf.create_simulation(sim_create_payload)
        sim_id = (
            sim_result.get("simulation_id") or sim_result.get("id")
            if isinstance(sim_result, dict)
            else sim_result
        )
        store.update(
            job_id,
            stage="generate_profiles",
            stages_completed=["ontology_generate", "graph_build", "simulation_create"],
            progress=42,
        )

        # Stage 4 – Generate profiles
        profiles_payload = {"simulation_id": sim_id, **params.get("profiles", {})}
        mf.generate_profiles(profiles_payload)
        store.update(
            job_id,
            stage="simulation_start",
            stages_completed=[
                "ontology_generate", "graph_build", "simulation_create",
                "generate_profiles",
            ],
            progress=56,
        )

        # Stage 5 – Start simulation
        start_payload = {"simulation_id": sim_id, **params.get("start", {})}
        mf.start_simulation(start_payload)
        store.update(
            job_id,
            stage="simulation_run",
            stages_completed=[
                "ontology_generate", "graph_build", "simulation_create",
                "generate_profiles", "simulation_start",
            ],
            progress=70,
        )

        # Stage 6 – Wait for simulation to finish
        run_result = mf.wait_for_simulation(sim_id, poll_interval=5.0, timeout=LLM_TIMEOUT)
        store.update(
            job_id,
            stage="report_generate",
            stages_completed=[
                "ontology_generate", "graph_build", "simulation_create",
                "generate_profiles", "simulation_start", "simulation_run",
            ],
            progress=85,
        )

        # Stage 7 – Generate report
        report_payload = {"simulation_id": sim_id, **params.get("report", {})}
        report_result = mf.generate_report(report_payload)

        store.update(
            job_id,
            status="completed",
            stage=None,
            stages_completed=PIPELINE_STAGES,
            progress=100,
            result={
                "simulation_id": sim_id,
                "ontology": ontology_result,
                "graph": graph_result,
                "simulation": sim_result,
                "run": run_result,
                "report": report_result,
            },
        )

    except Exception as exc:
        store.update(job_id, status="failed", error=str(exc))


# ======================================================================
# Flask application
# ======================================================================

app = Flask(__name__)
CORS(app)


# --- Health ---------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def api_health():
    """Proxy health-check: checks both this server and MiroFish."""
    mirofish_ok = False
    try:
        mf.health()
        mirofish_ok = True
    except Exception:
        pass

    return jsonify({
        "status": "ok",
        "mirofish_reachable": mirofish_ok,
        "mirofish_url": MIROFISH_BASE_URL,
    })


# --- Static data ----------------------------------------------------------

@app.route("/api/countries", methods=["GET"])
def api_countries():
    return jsonify(EU_COUNTRIES)


@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    jobs = store.list_all()
    return jsonify(jobs)


# --- Simulate -------------------------------------------------------------

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Start a full simulation pipeline for a given EU country.

    Expected JSON body:
        {
            "country_code": "DE",
            "scenario": "optional free-text scenario description",
            "params": { ... optional per-stage overrides ... }
        }
    """
    body = request.get_json(silent=True) or {}
    country_code = body.get("country_code", "").upper()

    if country_code not in _COUNTRY_BY_CODE:
        return jsonify({"error": f"Unknown country code: {country_code}"}), 400

    scenario = body.get("scenario", "")
    params = body.get("params", {})

    job = store.create(country_code, scenario, params)

    thread = threading.Thread(target=_run_pipeline, args=(job["id"],), daemon=True)
    thread.start()

    return jsonify({"job_id": job["id"], "status": "queued"}), 202


# --- Status / Results -----------------------------------------------------

@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id: str):
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "stages_completed": job["stages_completed"],
        "progress": job["progress"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "error": job["error"],
    })


@app.route("/api/results/<job_id>", methods=["GET"])
def api_results(job_id: str):
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "completed":
        return jsonify({
            "error": "Results not ready",
            "status": job["status"],
            "progress": job["progress"],
        }), 409

    return jsonify(job["result"])


# --- Chat -----------------------------------------------------------------

@app.route("/api/chat/<job_id>", methods=["POST"])
def api_chat(job_id: str):
    """Chat about a completed simulation's report.

    Expected JSON body:
        { "message": "user question about the report" }
    """
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Simulation not completed yet"}), 409

    body = request.get_json(silent=True) or {}
    message = body.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    sim_id = job["result"]["simulation_id"]
    chat_payload = {"simulation_id": sim_id, "message": message}
    try:
        reply = mf.report_chat(chat_payload)
    except MiroFishAPIError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify({"reply": reply})


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
