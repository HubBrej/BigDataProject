import boto3
import time
import statistics
import json
from encryptionFile import *
# Create SQS client
sqs = boto3.client('sqs')
s3 = boto3.client('s3')
buckets = s3.list_buckets()


queue_url = 'https://queue.amazonaws.com/895414063070/requestQueue'

filename='predict.csv'

encrypt_file_desc(filename, 'My sample CMK')

s3.upload_file(filename[:-len(filename.split('.')[-1])] + 'encrypted.'+filename.split('.')[-1], 'telecom-big-data', filename)

# Send message to SQS queue
response = sqs.send_message(
    QueueUrl=queue_url, MessageBody=(filename)
)

print(response["MessageId"])