"""
consumer.py
-----------
Background Kafka consumer for the 'prediction-events' topic.

Responsibilities:
  1. Continuously consume PredictionEvent messages.
  2. Persist each one to PostgreSQL via the repository layer (durable
     history, queryable by the API).
  3. Maintain an in-memory rolling window (deque) of recent feature
     vectors + probabilities for fast access by the drift scheduler —
     avoids hitting Postgres on every scheduled evaluation tick.

Run as a long-lived background thread/process (see scheduler.py, which
starts this alongside the APScheduler job).
"""

import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field

from kafka import KafkaConsumer
try:
    from kafka.errors import NoBrokersAvailable
except ImportError:
    # Some kafka-python forks/versions renamed or dropped this exception;
    # fall back to the generic KafkaError so error handling still works.
    from kafka.errors import KafkaError as NoBrokersAvailable

from src.monitoring.database import SessionLocal
from src.monitoring import repository as repo
from src.prediction.schemas import PredictionEvent

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC: str = os.getenv("KAFKA_TOPIC", "prediction-events")
KAFKA_CONSUMER_GROUP: str = os.getenv("KAFKA_CONSUMER_GROUP", "drift-consumer")

# Max number of recent events kept in memory for drift evaluation.
# Matches the design doc's recommended cap for real-time Evidently reports.
DRIFT_WINDOW_SIZE: int = int(os.getenv("DRIFT_WINDOW_SIZE", "1000"))

# How many events to batch before writing to Postgres in one transaction.
DB_FLUSH_BATCH_SIZE: int = int(os.getenv("DB_FLUSH_BATCH_SIZE", "50"))


@dataclass
class DriftWindow:
    """
    Thread-safe in-memory rolling window of recent prediction events.

    Used by the scheduler (Day 6, step 2) to snapshot the current state
    for drift evaluation without querying Postgres on every tick.
    """

    max_size: int = DRIFT_WINDOW_SIZE
    _features: deque = field(default_factory=deque, repr=False)
    _probabilities: deque = field(default_factory=deque, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, features: dict[str, float], probability: float) -> None:
        with self._lock:
            self._features.append(features)
            self._probabilities.append(probability)
            while len(self._features) > self.max_size:
                self._features.popleft()
                self._probabilities.popleft()

    def snapshot(self) -> tuple[list[dict], list[float]]:
        """
        Return a point-in-time copy of the window's contents.
        Safe to call concurrently with append() from the consumer thread.
        """
        with self._lock:
            return list(self._features), list(self._probabilities)

    def __len__(self) -> int:
        with self._lock:
            return len(self._features)


# Module-level singleton window, shared between the consumer thread and
# the scheduler's evaluation job.
drift_window = DriftWindow()


class DriftConsumer:
    """
    Wraps a KafkaConsumer subscribed to 'prediction-events'.

    Call run() in a background thread (see start_consumer_thread below).
    Each consumed message is:
      1. Parsed into a PredictionEvent.
      2. Appended to the in-memory drift_window.
      3. Buffered for batched insertion into PostgreSQL.
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        topic: str = KAFKA_TOPIC,
        group_id: str = KAFKA_CONSUMER_GROUP,
        flush_batch_size: int = DB_FLUSH_BATCH_SIZE,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.flush_batch_size = flush_batch_size
        self._stop_event = threading.Event()
        self._buffer: list[dict] = []

    def _build_consumer(self) -> KafkaConsumer:
        return KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers.split(","),
            group_id=self.group_id,
            auto_offset_reset="latest",   # only new events; historical replay is a separate concern
            enable_auto_commit=True,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            consumer_timeout_ms=1000,      # allows checking _stop_event periodically
        )

    def stop(self) -> None:
        logger.info("Stopping drift consumer …")
        self._stop_event.set()

    def run(self) -> None:
        """
        Main consume loop. Intended to run forever in a background thread
        until stop() is called. Reconnects with backoff if Kafka is
        temporarily unavailable.
        """
        logger.info(
            "Starting drift consumer — topic: %s | group: %s | brokers: %s",
            self.topic, self.group_id, self.bootstrap_servers,
        )

        consumer = None
        while not self._stop_event.is_set():
            try:
                if consumer is None:
                    consumer = self._build_consumer()
                    logger.info("Kafka consumer connected ✓")

                for message in consumer:
                    if self._stop_event.is_set():
                        break
                    self._handle_message(message.value)

            except NoBrokersAvailable:
                logger.warning("Kafka unavailable — retrying in 5s …")
                consumer = None
                self._stop_event.wait(timeout=5)
            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected consumer error: %s — retrying in 5s", exc)
                consumer = None
                self._stop_event.wait(timeout=5)

        self._flush_buffer()
        if consumer is not None:
            consumer.close()
        logger.info("Drift consumer stopped.")

    def _handle_message(self, raw_value: dict) -> None:
        try:
            event = PredictionEvent.model_validate(raw_value)
        except Exception as exc:
            logger.warning("Skipping malformed event: %s", exc)
            return

        # 1. Update the in-memory window immediately (cheap, no I/O)
        drift_window.append(event.features, event.probability)

        # 2. Buffer for batched DB insertion
        self._buffer.append(
            {
                "features": event.features,
                "prediction": event.prediction,
                "probability": event.probability,
                "created_at": event.timestamp,
            }
        )

        if len(self._buffer) >= self.flush_batch_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return

        db = SessionLocal()
        try:
            count = repo.bulk_create_prediction_records(db, self._buffer)
            logger.info("Flushed %d prediction records to Postgres", count)
            self._buffer.clear()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to flush prediction records: %s", exc)
            db.rollback()
        finally:
            db.close()


def start_consumer_thread() -> tuple[DriftConsumer, threading.Thread]:
    """
    Convenience helper — creates a DriftConsumer and starts it on a
    daemon thread. Returns both so the caller can call .stop() on
    shutdown.
    """
    consumer = DriftConsumer()
    thread = threading.Thread(target=consumer.run, daemon=True, name="drift-consumer")
    thread.start()
    return consumer, thread