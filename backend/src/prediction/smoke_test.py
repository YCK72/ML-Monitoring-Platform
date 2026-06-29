"""
smoke_test.py
-------------
End-to-end smoke test that:
  1. Sends 100 POST /predict requests with random feature vectors.
  2. Asserts every response is HTTP 200 with a valid probability in [0, 1].
  3. Reads 100 messages back from the 'prediction-events' Kafka topic and
     prints them to confirm end-to-end delivery.

Usage:
    python -m src.prediction.smoke_test
    python -m src.prediction.smoke_test --url http://localhost:8001 --count 100
"""

import argparse
import json
import logging
import time
import uuid

import numpy as np
import requests
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from src.training.generate_data import FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def random_feature_vector(rng: np.random.Generator) -> dict[str, float]:
    """Generate a random feature vector matching the training distribution."""
    return {name: float(rng.normal(0, 1)) for name in FEATURE_NAMES}


# ── HTTP smoke test ───────────────────────────────────────────────────────────

def run_http_smoke_test(
    base_url: str,
    count: int,
    seed: int,
) -> list[dict]:
    """
    Send *count* POST /predict requests and assert all succeed.

    Returns the list of response JSON dicts for inspection.

    Uses a single requests.Session so the underlying TCP connection is
    reused (keep-alive) across all requests, rather than reconnecting
    (and re-resolving the hostname) on every call. On Windows in
    particular, repeated fresh connections to 'localhost' can each pay
    an IPv6-then-IPv4-fallback delay; a persistent session avoids that
    entirely after the first connection.
    """
    rng = np.random.default_rng(seed)
    results = []
    errors = 0

    logger.info("Sending %d requests to %s/predict …", count, base_url)
    start = time.perf_counter()

    session = requests.Session()
    for i in range(count):
        payload = {"features": random_feature_vector(rng)}
        try:
            resp = session.post(f"{base_url}/predict", json=payload, timeout=5)

            # Assert HTTP 200
            assert resp.status_code == 200, (
                f"Request {i} failed: status={resp.status_code} body={resp.text}"
            )

            data = resp.json()

            # Assert response shape
            assert "prediction" in data,   f"Request {i}: missing 'prediction'"
            assert "probability" in data,  f"Request {i}: missing 'probability'"
            assert data["prediction"] in (0, 1), (
                f"Request {i}: prediction={data['prediction']} not in {{0, 1}}"
            )
            assert 0.0 <= data["probability"] <= 1.0, (
                f"Request {i}: probability={data['probability']} out of [0, 1]"
            )

            results.append(data)

        except requests.exceptions.ConnectionError:
            logger.error(
                "Connection refused at %s — is the prediction service running?",
                base_url,
            )
            raise
        except AssertionError as exc:
            logger.error("Assertion failed: %s", exc)
            errors += 1

    session.close()
    elapsed = time.perf_counter() - start
    rps = count / elapsed

    logger.info(
        "HTTP smoke test complete — %d/%d passed | %.1f req/s | %.0f ms total",
        count - errors, count, rps, elapsed * 1000,
    )

    if errors:
        raise RuntimeError(f"{errors} requests failed — see logs above.")

    return results


# ── Kafka consumer verification ───────────────────────────────────────────────

def verify_kafka_events(
    brokers: str,
    topic: str,
    expected_count: int,
    timeout_seconds: float = 30.0,
) -> list[dict]:
    """
    Spin up a temporary consumer from the earliest offset and collect
    *expected_count* messages (or whatever arrives within *timeout_seconds*).

    Prints each message and returns the list of dicts.
    """
    logger.info(
        "Reading from Kafka topic '%s' (brokers: %s) …", topic, brokers
    )

    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=brokers.split(","),
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            consumer_timeout_ms=int(timeout_seconds * 1000),
            group_id=f"smoke-test-{uuid.uuid4().hex[:8]}",  # unique group = fresh read
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        )
    except NoBrokersAvailable:
        logger.warning(
            "Kafka not reachable at %s — skipping Kafka verification.", brokers
        )
        return []

    messages = []
    for msg in consumer:
        messages.append(msg.value)
        if len(messages) >= expected_count:
            break

    consumer.close()

    logger.info(
        "Kafka verification — received %d/%d messages",
        len(messages), expected_count,
    )

    if messages:
        logger.info("Sample message:\n%s", json.dumps(messages[0], indent=2, default=str))

    return messages


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end smoke test for the prediction service."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8001",
        help="Base URL of the prediction service (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--brokers",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    parser.add_argument(
        "--topic",
        default="prediction-events",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of prediction requests to send (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--skip-kafka",
        action="store_true",
        help="Skip the Kafka consumer verification step.",
    )
    args = parser.parse_args()

    # Step 1: HTTP smoke test
    run_http_smoke_test(
        base_url=args.url,
        count=args.count,
        seed=args.seed,
    )

    # Step 2: Kafka verification
    if not args.skip_kafka:
        verify_kafka_events(
            brokers=args.brokers,
            topic=args.topic,
            expected_count=args.count,
        )

    logger.info("All smoke tests passed ✓")


if __name__ == "__main__":
    main()