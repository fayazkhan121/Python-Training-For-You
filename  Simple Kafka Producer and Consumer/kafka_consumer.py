from confluent_kafka import Consumer, KafkaError
import json

# Kafka consumer configuration
conf = {
    'bootstrap.servers': 'localhost:9092',  # Change this to your Kafka broker address
    'group.id': 'python-consumer-group',
    'auto.offset.reset': 'earliest'  # Start reading from the beginning of the topic
}

# Create Consumer instance
consumer = Consumer(conf)


def consume_messages(topic):
    """Consumes messages from the specified topic"""
    try:
        # Subscribe to topic
        consumer.subscribe([topic])

        while True:
            # Poll for messages
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print('Reached end of partition')
                else:
                    print(f'Error: {msg.error()}')
            else:
                try:
                    # Decode and parse the message
                    message = msg.value().decode('utf-8')
                    # Try to parse as JSON
                    try:
                        message = json.loads(message)
                    except json.JSONDecodeError:
                        pass  # Keep message as string if not JSON

                    print(f'Received message: {message}')
                except Exception as e:
                    print(f'Error processing message: {e}')

    except KeyboardInterrupt:
        print('Stopping consumer...')
    finally:
        # Close consumer to commit final offsets
        consumer.close()


if __name__ == '__main__':
    topic = 'test-topic'  # Change this to match your producer's topic
    consume_messages(topic)