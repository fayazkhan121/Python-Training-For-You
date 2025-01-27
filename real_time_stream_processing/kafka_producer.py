import json
import random
import time
from kafka import KafkaProducer

TOPIC = "transactions"
BROKER = "localhost:9092"

def generate_transaction():
    return {
        "user_id": f"user_{random.randint(1, 100)}",
        "transaction_id": f"txn_{random.randint(10000, 99999)}",
        "amount": round(random.uniform(10, 1000), 2),
        "timestamp": int(time.time())
    }

def produce_transactions():
    producer = KafkaProducer(
        bootstrap_servers=BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    while True:
        transaction = generate_transaction()
        print(f"Sending: {transaction}")
        producer.send(TOPIC, value=transaction)
        time.sleep(random.uniform(0.1, 1))  # Random delay between events

if __name__ == "__main__":
    produce_transactions()
