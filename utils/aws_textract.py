import boto3
from PIL import Image
import io
import os

def process_text_detection(bucket, image_name, region):

    extractions = dict()

    # Get credentials from environment variables for security
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    #Get the S3
    s3 = session.resource('s3')

    # upload document in s3             
    data = open(image_name, 'rb')
    s3.Bucket(bucket).put_object(Key=image_name, Body=data)
    
    s3_object = s3.Object(bucket,image_name)
    s3_response = s3_object.get()

    stream = io.BytesIO(s3_response['Body'].read())
    image=Image.open(stream)

   
    # Detect text in the document
    client = session.client('textract', region_name = region)

    #process using S3 object
    response = client.detect_document_text(
        Document={'S3Object': {'Bucket': bucket, 'Name': image_name}})

    #Get the text blocks
    blocks=response['Blocks']   
    print ('Detected Document Text')
   
    # Create image showing bounding box/polygon the detected lines/text
    for block in blocks:
        if block['BlockType'] != 'PAGE':
                extractions[block['Text']] = block['Confidence']
    
    s = dict(reversed(sorted(extractions.items(), key=lambda item: item[1])))

    for k, v in s.items():
        if(len(k) > 5):
            return k
