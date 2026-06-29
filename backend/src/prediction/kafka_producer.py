import logging
import os
from typing import Optional

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

from src.prediction.schemas import PredictionEvent

logger = logging.getLogger(__name__)

# ── Config from environment ───────────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS: str = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
)
KAFKA_TOPIC: str = os.getenv("KAFKA_TOPIC", "prediction-events")

# ── Singleton state ───────────────────────────────────────────────────────────

_producer: Optional[KafkaProducer] = None


def get_producer() -> KafkaProducer:
    """
    Return the module-level KafkaProducer, creating it on first call.

    Raises NoBrokersAvailable if Kafka is not reachable at startup.
    """
    global _producer
    if _producer is None:
        logger.info(
            "Initialising Kafka producer — brokers: %s",
            KAFKA_BOOTSTRAP_SERVERS,
        )
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
            value_serializer=lambda v: v.encode("utf-8"),
            acks="all",
            retries=3,
            retry_backoff_ms=200,
            request_timeout_ms=10_000,
            max_block_ms=5_000,   # fail fast if broker is unreachable
        )
        logger.info("Kafka producer ready ✓")
    return _producer


def publish_prediction(event: PredictionEvent) -> None:
    """
    Publish a PredictionEvent to the Kafka topic.

    This is intentionally fire-and-forget: the Future returned by
    producer.send() is not awaited, so this function returns immediately
    and never blocks the HTTP response path.

    Errors are caught and logged — a Kafka outage must never crash
    the prediction service.

    Parameters
    ----------
    event:
        A fully populated PredictionEvent instance.
    """
    try:
        producer = get_producer()
        future = producer.send(
            topic=KAFKA_TOPIC,
            value=event.to_json(),
            # Use model_name as the partition key so all events for the
            # same model land on the same partition (ordered processing).
            key=event.model_name.encode("utf-8"),
        )

        # Attach non-blocking callbacks for observability
        future.add_callback(_on_send_success)
        future.add_errback(_on_send_error)

    except NoBrokersAvailable:
        logger.error(
            "Kafka broker unavailable — prediction event NOT published. "
            "Check KAFKA_BOOTSTRAP_SERVERS=%s",
            KAFKA_BOOTSTRAP_SERVERS,
        )
    except KafkaError as exc:
        logger.error("Kafka error publishing event: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error publishing to Kafka: %s", exc)


def _on_send_success(metadata: object) -> None:
    logger.debug(
        "Event published — topic: %s | partition: %s | offset: %s",
        getattr(metadata, "topic", "?"),
        getattr(metadata, "partition", "?"),
        getattr(metadata, "offset", "?"),
    )


def _on_send_error(exc: Exception) -> None:
    logger.error("Failed to deliver Kafka message: %s", exc)


def close_producer() -> None:
    """
    Flush pending messages and close the producer cleanly.
    Call this during application shutdown (FastAPI lifespan event).
    """
    global _producer
    if _producer is not None:
        logger.info("Flushing and closing Kafka producer …")
        _producer.flush(timeout=10)
        _producer.close()
        _producer = None
        logger.info("Kafka producer closed ✓")


def is_connected() -> bool:
    """
    Return True if the producer can reach at least one broker.
    Used by the /health endpoint.
    """
    try:
        producer = get_producer()
        # bootstrap_connected() is a lightweight check with no network I/O
        return producer.bootstrap_connected()
    except Exception:
        return False