import boto3
import time
import statistics
import json

# Create SQS client
sqs = boto3.client("sqs")
s3 = boto3.client("s3")

queue_response = "https://queue.amazonaws.com/895414063070/responseQueue"
# Receive message from SQS queue
while True:
    response = sqs.receive_message(
        QueueUrl=queue_response,
        AttributeNames=["SentTimestamp"],
        MaxNumberOfMessages=1,
        MessageAttributeNames=["All"],
        VisibilityTimeout=10,
        WaitTimeSeconds=0,
    )
    if "Messages" in response.keys():
        message = response["Messages"][0]
        receipt_handle = message["ReceiptHandle"]

        # Delete received message from queue
        sqs.delete_message(QueueUrl=queue_response, ReceiptHandle=receipt_handle)
        filename = message["Body"]
        s3.download_file("telecom-big-data", filename, filename)
        
        s3.delete_object(Bucket='telecom-big-data', Key=filename)

    time.sleep(1)
