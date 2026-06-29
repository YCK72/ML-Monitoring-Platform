"""
setup_kafka.py
--------------
Creates the 'prediction-events' Kafka topic if it doesn't already exist.

Run this once before starting the prediction service or drift consumer.
In Docker Compose it is invoked as an init step in the prediction service
entrypoint, after the Kafka healthcheck passes.

Usage:
    python -m src.prediction.setup_kafka
    python -m src.prediction.setup_kafka --brokers kafka:29092 --partitions 3
"""

import argparse
import logging
import os
import time

from kafka import KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import KafkaError, TopicAlreadyExistsError, NoBrokersAvailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BROKERS    = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEFAULT_TOPIC      = os.getenv("KAFKA_TOPIC", "prediction-events")
DEFAULT_PARTITIONS = int(os.getenv("KAFKA_PARTITIONS", "3"))
DEFAULT_REPLICATION = int(os.getenv("KAFKA_REPLICATION_FACTOR", "1"))


def create_topic(
    brokers: str,
    topic: str,
    num_partitions: int,
    replication_factor: int,
    retries: int = 10,
    retry_delay: float = 3.0,
) -> None:
    """
    Create *topic* on the Kafka cluster reachable at *brokers*.

    Retries up to *retries* times with *retry_delay* seconds between
    attempts — necessary because Kafka takes a few seconds to become
    ready after Docker Compose starts Zookeeper.

    Silently succeeds if the topic already exists.
    """
    broker_list = brokers.split(",")

    for attempt in range(1, retries + 1):
        try:
            admin = KafkaAdminClient(
                bootstrap_servers=broker_list,
                client_id="setup-kafka",
                request_timeout_ms=10_000,
            )
            break
        except NoBrokersAvailable:
            if attempt == retries:
                raise
            logger.warning(
                "Kafka not reachable yet (attempt %d/%d) — retrying in %.0fs …",
                attempt, retries, retry_delay,
            )
            time.sleep(retry_delay)

    # Check if topic already exists
    existing = admin.list_topics()
    if topic in existing:
        logger.info("Topic '%s' already exists — nothing to do.", topic)
        admin.close()
        return

    # Create the topic
    new_topic = NewTopic(
        name=topic,
        num_partitions=num_partitions,
        replication_factor=replication_factor,
        topic_configs={
            # Keep prediction events for 7 days (replaying drift windows)
            "retention.ms": str(7 * 24 * 60 * 60 * 1000),
            # Compact + delete policy for long-term storage efficiency
            "cleanup.policy": "delete",
        },
    )

    try:
        admin.create_topics([new_topic], validate_only=False)
        logger.info(
            "Topic '%s' created — partitions: %d | replication: %d",
            topic, num_partitions, replication_factor,
        )
    except TopicAlreadyExistsError:
        logger.info("Topic '%s' already exists (race condition) — OK.", topic)
    except KafkaError as exc:
        logger.error("Failed to create topic '%s': %s", topic, exc)
        raise
    finally:
        admin.close()


def verify_topic(brokers: str, topic: str) -> bool:
    """Return True if *topic* exists on the cluster."""
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=brokers.split(","),
            client_id="setup-kafka-verify",
        )
        exists = topic in admin.list_topics()
        admin.close()
        return exists
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the prediction-events Kafka topic."
    )
    parser.add_argument("--brokers", default=DEFAULT_BROKERS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--partitions", type=int, default=DEFAULT_PARTITIONS)
    parser.add_argument("--replication", type=int, default=DEFAULT_REPLICATION)
    parser.add_argument(
        "--retries", type=int, default=10,
        help="Number of connection retries before giving up (default: 10)"
    )
    args = parser.parse_args()

    create_topic(
        brokers=args.brokers,
        topic=args.topic,
        num_partitions=args.partitions,
        replication_factor=args.replication,
        retries=args.retries,
    )

    if verify_topic(args.brokers, args.topic):
        logger.info("Verification passed — topic '%s' is live ✓", args.topic)
    else:
        logger.error("Verification FAILED — topic '%s' not found after creation", args.topic)


if __name__ == "__main__":
    main()