import boto3
import time
import statistics
import json
from PIL import Image
import numpy as np
from predict import *
from keras.models import load_model
from encryptionFile import *
# Create SQS client
sqs = boto3.client('sqs')
s3 = boto3.client('s3')

saved_model = load_model('best_model.h5')

queue_url = 'https://queue.amazonaws.com/895414063070/requestQueue'
queue_response='https://sqs.us-east-1.amazonaws.com/895414063070/response_queue'
# Receive message from SQS queue
while True:
    response = sqs.receive_message(
        QueueUrl=queue_url,
        AttributeNames=[
            'SentTimestamp'
        ],
        MaxNumberOfMessages=1,
        MessageAttributeNames=[
            'All'
        ],
        VisibilityTimeout=10,
        WaitTimeSeconds=0
    )
    if 'Messages' in response.keys():
        message = response['Messages'][0]
        receipt_handle = message['ReceiptHandle']

        #Delete received message from queue
        sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )
        filename=message['Body']
        name=filename.split('.')[0]
        print(filename)
        s3.download_file('telecom-big-data', filename, 'temp.encrypted.csv')

        decrypt_file('temp.csv')
        
        preds=predict_cl('temp.decrypted.csv', saved_model) 
        
        s3.delete_object(Bucket='telecom-big-data', Key=filename)
        
        encrypt_file_desc('results.csv', 'My sample CMK')

        s3.upload_file('results.encrypted.csv', 'telecom-big-data', 'results.csv')
        # Send response
        response = sqs.send_message(
            QueueUrl=queue_response, MessageBody=('results.csv')
        )


    time.sleep(3)
