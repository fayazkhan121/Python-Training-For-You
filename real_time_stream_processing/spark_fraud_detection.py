from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, sum as _sum
from pyspark.sql.types import StructType, StringType, FloatType, IntegerType

BROKER = "localhost:9092"
INPUT_TOPIC = "transactions"
OUTPUT_TOPIC = "fraud_alerts"

# Define the schema for incoming transaction data
schema = StructType() \
    .add("user_id", StringType()) \
    .add("transaction_id", StringType()) \
    .add("amount", FloatType()) \
    .add("timestamp", IntegerType())

# Spark session
spark = SparkSession.builder \
    .appName("FraudDetection") \
    .getOrCreate()

# Read stream from Kafka
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", BROKER) \
    .option("subscribe", INPUT_TOPIC) \
    .load()

# Parse JSON and extract fields
parsed_stream = raw_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Process data to detect fraud
fraudulent_users = parsed_stream \
    .withWatermark("timestamp", "1 minute") \
    .groupBy(
        col("user_id"),
        window(col("timestamp").cast("timestamp"), "1 minute", "30 seconds")
    ) \
    .agg(_sum("amount").alias("total_amount")) \
    .filter(col("total_amount") > 500)  # Fraud detection threshold

# Write fraudulent activities to Kafka
fraud_alerts = fraudulent_users.selectExpr(
    "CAST(user_id AS STRING) AS key",
    "to_json(struct(*)) AS value"
)

query = fraud_alerts.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", BROKER) \
    .option("topic", OUTPUT_TOPIC) \
    .option("checkpointLocation", "/tmp/checkpoints/fraud_detection") \
    .start()

query.awaitTermination()
