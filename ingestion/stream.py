import boto3
import json
import redis

# AWS SQS Client
QUEUE_URL = os.getenv('AWS_SQS_QUEUE_URL')
REGION_NAME = os.getenv('AWS_REGION')

# Initialize client
sqs = boto3.client('sqs', region_name=REGION_NAME)

def poll_stream():
    """Poll SQS for new image processing jobs using environment-configured URL."""
    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20
        )
        
        if 'Messages' in response:
            for message in response['Messages']:
                data = json.loads(message['Body'])
                # Push job ID to Redis for the InferenceWorker
                r.lpush("inference_queue", data['job_id'])
                
                # Delete message from SQS after successful queueing
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=message['ReceiptHandle']
                )