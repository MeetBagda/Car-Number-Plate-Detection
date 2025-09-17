#!/usr/bin/env python3
"""
Script to create the required S3 bucket for the license plate detection app.
"""
import boto3
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def create_s3_bucket():
    try:
        # Get credentials from environment
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if not aws_access_key_id or not aws_secret_access_key:
            print("❌ AWS credentials not found in environment variables!")
            return False
            
        # Create session
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # Create S3 client for ap-south-1 region
        s3_client = session.client('s3', region_name='ap-south-1')
        
        bucket_name = 'car-plate-extractor-meet4224-2025'
        region = 'ap-south-1'
        
        # Create bucket with location constraint for regions other than us-east-1
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': region}
        )
        
        print(f"✅ Successfully created S3 bucket: {bucket_name} in region: {region}")
        return True
        
    except Exception as e:
        if "BucketAlreadyOwnedByYou" in str(e):
            print(f"✅ S3 bucket '{bucket_name}' already exists and is owned by you!")
            return True
        else:
            print(f"❌ Error creating S3 bucket: {e}")
            return False

if __name__ == "__main__":
    print("Creating S3 bucket for license plate detection app...")
    create_s3_bucket()
