from confluent_kafka import Producer
import json
import time

# Kafka broker configuration
conf = {
    'bootstrap.servers': 'localhost:9092',  # Change this to your Kafka broker address
    'client.id': 'python-producer'
}

# Create Producer instance
producer = Producer(conf)


def delivery_report(err, msg):
    """Callback function to handle delivery reports"""
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}')


def produce_message(topic, message):
    """Produces a message to the specified topic"""
    try:
        # Convert message to JSON string if it's a dictionary
        if isinstance(message, dict):
            message = json.dumps(message)

        # Produce message
        producer.produce(
            topic=topic,
            value=message.encode('utf-8'),
            callback=delivery_report
        )
        # Flush to ensure message is sent
        producer.flush()

    except Exception as e:
        print(f'Error producing message: {e}')


if __name__ == '__main__':
    topic = 'test-topic'  # Change this to your desired topic name

    # Example: Send 5 test messages
    for i in range(5):
        message = {
            'message_id': i,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'content': f'Test message {i}'
        }
        produce_message(topic, message)
        time.sleep(1)  # Wait for 1 second between messages