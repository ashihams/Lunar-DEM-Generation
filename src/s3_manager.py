"""
S3 Data Lake Manager for Lunar DEM Pipeline
Handles data movement between S3 buckets (raw, staging, processed)
"""
import os
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3DataLakeManager:
    """Manages data flow through S3-based data lake architecture"""
    
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1"
    ):
        """
        Initialize S3 client.
        If credentials not provided, will use AWS CLI config or IAM role.
        """
        session_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        
        self.s3_client = boto3.client('s3', **session_kwargs)
        self.s3_resource = boto3.resource('s3', **session_kwargs)
        
    def create_bucket_if_not_exists(self, bucket_name: str, region: str = "us-east-1"):
        """Create S3 bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} already exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    if region == "us-east-1":
                        self.s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': region}
                        )
                    logger.info(f"Created bucket: {bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Error creating bucket: {create_error}")
                    raise
            else:
                logger.error(f"Error checking bucket: {e}")
                raise
    
    def upload_file(
        self, 
        local_path: str, 
        bucket: str, 
        s3_key: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Upload a file to S3"""
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = {k: str(v) for k, v in metadata.items()}
            
            self.s3_client.upload_file(local_path, bucket, s3_key, ExtraArgs=extra_args)
            s3_uri = f"s3://{bucket}/{s3_key}"
            logger.info(f"Uploaded {local_path} to {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Error uploading file: {e}")
            raise
    
    def upload_directory(
        self, 
        local_dir: str, 
        bucket: str, 
        s3_prefix: str,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[str]:
        """Upload entire directory to S3"""
        uploaded_files = []
        local_path = Path(local_dir)
        
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                # Check exclusion patterns
                if exclude_patterns:
                    if any(pattern in str(file_path) for pattern in exclude_patterns):
                        continue
                
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative_path}".replace("\\", "/")
                
                s3_uri = self.upload_file(str(file_path), bucket, s3_key)
                uploaded_files.append(s3_uri)
        
        logger.info(f"Uploaded {len(uploaded_files)} files from {local_dir}")
        return uploaded_files
    
    def download_file(self, bucket: str, s3_key: str, local_path: str) -> str:
        """Download a file from S3"""
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(bucket, s3_key, local_path)
            logger.info(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Error downloading file: {e}")
            raise
    
    def download_directory(
        self, 
        bucket: str, 
        s3_prefix: str, 
        local_dir: str
    ) -> List[str]:
        """Download all files under S3 prefix to local directory"""
        downloaded_files = []
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    # Skip directory markers
                    if s3_key.endswith('/'):
                        continue
                    
                    relative_path = s3_key[len(s3_prefix):].lstrip('/')
                    local_path = os.path.join(local_dir, relative_path)
                    
                    self.download_file(bucket, s3_key, local_path)
                    downloaded_files.append(local_path)
            
            logger.info(f"Downloaded {len(downloaded_files)} files to {local_dir}")
            return downloaded_files
        except ClientError as e:
            logger.error(f"Error downloading directory: {e}")
            raise
    
    def list_objects(self, bucket: str, prefix: str = "", max_keys: int = 1000) -> List[Dict]:
        """List objects in S3 bucket with prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    objects.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'etag': obj['ETag']
                    })
            
            logger.info(f"Found {len(objects)} objects in s3://{bucket}/{prefix}")
            return objects
        except ClientError as e:
            logger.error(f"Error listing objects: {e}")
            raise
    
    def copy_object(
        self, 
        source_bucket: str, 
        source_key: str, 
        dest_bucket: str, 
        dest_key: str
    ) -> str:
        """Copy object from one S3 location to another"""
        try:
            copy_source = {'Bucket': source_bucket, 'Key': source_key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_key
            )
            dest_uri = f"s3://{dest_bucket}/{dest_key}"
            logger.info(f"Copied s3://{source_bucket}/{source_key} to {dest_uri}")
            return dest_uri
        except ClientError as e:
            logger.error(f"Error copying object: {e}")
            raise
    
    def delete_object(self, bucket: str, key: str):
        """Delete object from S3"""
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Deleted s3://{bucket}/{key}")
        except ClientError as e:
            logger.error(f"Error deleting object: {e}")
            raise
    
    def save_metadata(
        self, 
        metadata: Dict, 
        bucket: str, 
        key: str
    ) -> str:
        """Save metadata as JSON to S3"""
        try:
            json_data = json.dumps(metadata, indent=2)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            s3_uri = f"s3://{bucket}/{key}"
            logger.info(f"Saved metadata to {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Error saving metadata: {e}")
            raise
    
    def load_metadata(self, bucket: str, key: str) -> Dict:
        """Load metadata JSON from S3"""
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"Loaded metadata from s3://{bucket}/{key}")
            return metadata
        except ClientError as e:
            logger.error(f"Error loading metadata: {e}")
            raise


class DataLakePipeline:
    """
    Orchestrates data flow through data lake layers:
    - Raw Layer: s3://lunar-dem-raw/
    - Staging Layer: s3://lunar-dem-staging/
    - Processed Layer: s3://lunar-dem-processed/
    """
    
    def __init__(
        self,
        raw_bucket: str = "lunar-dem-raw",
        staging_bucket: str = "lunar-dem-staging",
        processed_bucket: str = "lunar-dem-processed",
        region: str = "us-east-1"
    ):
        self.raw_bucket = raw_bucket
        self.staging_bucket = staging_bucket
        self.processed_bucket = processed_bucket
        self.s3_manager = S3DataLakeManager(region_name=region)
        
        # Create buckets if they don't exist
        for bucket in [raw_bucket, staging_bucket, processed_bucket]:
            self.s3_manager.create_bucket_if_not_exists(bucket, region)
    
    def ingest_raw_data(
        self, 
        local_dir: str, 
        crater_name: str,
        run_id: str
    ) -> Dict:
        """
        Stage 1: Ingest raw data from local to S3 raw layer
        """
        s3_prefix = f"raw/{crater_name}/{run_id}"
        
        logger.info(f"Ingesting raw data for {crater_name} (run: {run_id})")
        uploaded_files = self.s3_manager.upload_directory(
            local_dir, 
            self.raw_bucket, 
            s3_prefix
        )
        
        metadata = {
            "crater": crater_name,
            "run_id": run_id,
            "stage": "raw",
            "file_count": len(uploaded_files),
            "files": uploaded_files
        }
        
        metadata_key = f"{s3_prefix}/metadata.json"
        self.s3_manager.save_metadata(metadata, self.raw_bucket, metadata_key)
        
        return metadata
    
    def promote_to_staging(
        self, 
        crater_name: str, 
        run_id: str,
        preprocessed_local_dir: str
    ) -> Dict:
        """
        Stage 2: Upload preprocessed data to staging layer
        """
        s3_prefix = f"staging/{crater_name}/{run_id}"
        
        logger.info(f"Promoting preprocessed data to staging (run: {run_id})")
        uploaded_files = self.s3_manager.upload_directory(
            preprocessed_local_dir,
            self.staging_bucket,
            s3_prefix
        )
        
        metadata = {
            "crater": crater_name,
            "run_id": run_id,
            "stage": "staging",
            "file_count": len(uploaded_files),
            "files": uploaded_files
        }
        
        metadata_key = f"{s3_prefix}/metadata.json"
        self.s3_manager.save_metadata(metadata, self.staging_bucket, metadata_key)
        
        return metadata
    
    def publish_results(
        self,
        crater_name: str,
        run_id: str,
        results_local_dir: str
    ) -> Dict:
        """
        Stage 3: Publish final DEM results to processed layer
        """
        s3_prefix = f"processed/{crater_name}/{run_id}"
        
        logger.info(f"Publishing final results (run: {run_id})")
        uploaded_files = self.s3_manager.upload_directory(
            results_local_dir,
            self.processed_bucket,
            s3_prefix
        )
        
        metadata = {
            "crater": crater_name,
            "run_id": run_id,
            "stage": "processed",
            "file_count": len(uploaded_files),
            "files": uploaded_files
        }
        
        metadata_key = f"{s3_prefix}/metadata.json"
        self.s3_manager.save_metadata(metadata, self.processed_bucket, metadata_key)
        
        return metadata