# Kafka Producer-Consumer Example

This project demonstrates a simple implementation of a Kafka Producer and Consumer using Python and the confluent-kafka library.

## Prerequisites

- Python 3.6 or higher
- Apache Kafka broker running
- confluent-kafka Python package

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd kafka-producer-consumer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install confluent-kafka
```

## Project Structure

```
kafka-producer-consumer/
├── kafka_producer.py
├── kafka_consumer.py
└── README.md
```

## Configuration

Both the producer and consumer are configured to connect to a Kafka broker running on `localhost:9092` by default. You can modify this and other configuration parameters in the respective files:

### Producer Configuration
```python
conf = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'python-producer'
}
```

### Consumer Configuration
```python
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'python-consumer-group',
    'auto.offset.reset': 'earliest'
}
```

## Usage

1. Make sure your Kafka broker is running.

2. Create a Kafka topic (if it doesn't exist):
```bash
kafka-topics.sh --create --topic test-topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

3. Start the consumer in one terminal:
```bash
python kafka_consumer.py
```

4. Start the producer in another terminal:
```bash
python kafka_producer.py
```

## Features

### Producer (`kafka_producer.py`)
- Delivery confirmation callback for message tracking
- Support for both string and JSON message formats
- Comprehensive error handling
- Message flushing to ensure delivery
- Configurable topic and message content
- Built-in delivery report callback

### Consumer (`kafka_consumer.py`)
- Continuous message polling with error handling
- Support for both string and JSON message formats
- Graceful shutdown mechanism (using Ctrl+C)
- Configurable consumer group and offset reset behavior
- Automatic JSON parsing for structured messages
- Error handling for message processing

## Example Output

### Producer Output
```
Message delivered to test-topic [0] at offset 0
Message delivered to test-topic [0] at offset 1
Message delivered to test-topic [0] at offset 2
```

### Consumer Output
```
Received message: {'message_id': 0, 'timestamp': '2025-01-29 10:30:00', 'content': 'Test message 0'}
Received message: {'message_id': 1, 'timestamp': '2025-01-29 10:30:01', 'content': 'Test message 1'}
Received message: {'message_id': 2, 'timestamp': '2025-01-29 10:30:02', 'content': 'Test message 2'}
```

## Customization

You can customize various aspects of the implementation:

1. Message Format
   - Modify the message structure in `kafka_producer.py`
   - Adjust message processing in `kafka_consumer.py`

2. Configuration
   - Change broker address
   - Modify consumer group settings
   - Adjust polling timeout values
   - Configure batch sizes and buffer memory

3. Topic Management
   - Change topic names
   - Add multiple topic support
   - Implement topic creation logic

## Error Handling

The implementation includes error handling for common scenarios:
- Connection failures
- Message delivery failures
- Message processing errors
- JSON parsing errors
- Graceful shutdown

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Troubleshooting

1. Connection Issues
   - Verify Kafka broker is running
   - Check broker address and port
   - Ensure topic exists

2. Consumer Issues
   - Verify consumer group ID
   - Check offset reset configuration
   - Monitor consumer lag

3. Producer Issues
   - Check delivery reports
   - Verify message format
   - Monitor buffer memory

## Additional Resources

- [Confluent Kafka Python Documentation](https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Kafka Python Client Examples](https://github.com/confluentinc/confluent-kafka-python/tree/master/examples)
