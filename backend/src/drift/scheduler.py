"""
scheduler.py
------------
Starts the Kafka consumer and an APScheduler background job that calls
run_drift_evaluation() on a configurable interval.

Run standalone as the drift-scheduler service:
    python -m src.drift.scheduler

In Docker Compose, this is its own container — see Day 6, Step 4.
"""

import logging
import os
import signal
import sys
import time

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from src.drift.consumer import start_consumer_thread
from src.drift.evaluator import run_drift_evaluation
from src.monitoring.database import DATABASE_URL

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DRIFT_EVAL_INTERVAL_MINUTES: int = int(os.getenv("DRIFT_EVAL_INTERVAL_MINUTES", "5"))


def build_scheduler() -> BackgroundScheduler:
    """
    Configure APScheduler with a SQLAlchemy job store, so scheduled jobs
    survive process restarts (per the design doc's reliability requirement).
    """
    jobstores = {
        "default": SQLAlchemyJobStore(url=DATABASE_URL, tablename="apscheduler_jobs")
    }
    executors = {"default": ThreadPoolExecutor(max_workers=2)}

    scheduler = BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults={"coalesce": True, "max_instances": 1},
        timezone="UTC",
    )
    return scheduler


def _run_evaluation_job() -> None:
    """Wrapper so exceptions in the evaluation never kill the scheduler thread."""
    try:
        result = run_drift_evaluation()
        if result is not None:
            logger.info(
                "Scheduled evaluation complete — report #%s, severity: %s",
                result["id"], result["overall_severity"],
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("Scheduled drift evaluation raised an exception: %s", exc, exc_info=True)


def main() -> None:
    logger.info(
        "Starting drift scheduler service — eval interval: %d minute(s)",
        DRIFT_EVAL_INTERVAL_MINUTES,
    )

    # 1. Start the Kafka consumer in a background thread
    consumer, consumer_thread = start_consumer_thread()
    logger.info("Kafka consumer thread started ✓")

    # 2. Start the APScheduler job
    scheduler = build_scheduler()
    scheduler.add_job(
        _run_evaluation_job,
        trigger="interval",
        minutes=DRIFT_EVAL_INTERVAL_MINUTES,
        id="drift_evaluation_job",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("APScheduler started ✓ — job will run every %d minute(s)", DRIFT_EVAL_INTERVAL_MINUTES)

    # 3. Graceful shutdown on SIGINT/SIGTERM
    def _shutdown(signum, _frame):
        logger.info("Received signal %s — shutting down …", signum)
        scheduler.shutdown(wait=False)
        consumer.stop()
        consumer_thread.join(timeout=10)
        logger.info("Shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 4. Block forever (this is a long-running service process)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)


if __name__ == "__main__":
    main()