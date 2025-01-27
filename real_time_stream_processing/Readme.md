# Real-Time Stream Processing with Kafka and Spark

### Imagine you're building a real-time fraud detection system for online transactions. Events (transactions) are sent to Kafka, and Spark processes these streams to identify suspicious activities (e.g., transactions exceeding a specific threshold in a short time window). The system outputs detected fraud cases to another Kafka topic.


## How to Run the Code

### Step 1: Set up Kafka
1. Start **Kafka** and **Zookeeper**.
2. Create Kafka topics:
   ```bash
   kafka-topics.sh --create --topic transactions --bootstrap-server localhost:9092
   kafka-topics.sh --create --topic fraud_alerts --bootstrap-server localhost:9092
   ```

### Step 2: Install Dependencies
Install the required Python libraries by running the following command:
```bash
pip install -r requirements.txt
```

### Step 3: Start Kafka Producer
Run the producer to generate transaction events:
```bash
python kafka_producer.py
```

### Step 4: Start Spark Application
Run the Spark Structured Streaming application to process the data:
```bash
spark-submit spark_fraud_detection.py
```

### Step 5: Consume Fraud Alerts
To view the fraud alerts, consume messages from the `fraud_alerts` topic:
```bash
kafka-console-consumer.sh --topic fraud_alerts --bootstrap-server localhost:9092
```

---

## Advanced Features in the Code

### 1. **Sliding Window Aggregation**
- Transactions are aggregated over a **1-minute window** with a **30-second slide** to detect patterns.

### 2. **Watermarking**
- The system handles late data by discarding records outside a **1-minute watermark**.

### 3. **Real-Time Fraud Alerts**
- Fraudulent activities are published to a separate Kafka topic (`fraud_alerts`) for downstream processing or alerting.

---

## Project Structure
```plaintext
real_time_stream_processing/
├── kafka_producer.py        # Produces transactions to Kafka topic
├── spark_fraud_detection.py # Spark Structured Streaming logic
└── requirements.txt         # Python dependencies
```

---

Feel free to extend the code with additional features or integrate it with other services!
