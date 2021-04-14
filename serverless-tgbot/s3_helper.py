import os
import sys
import boto3
from botocore.exceptions import ClientError

class S3Helper():
    
    @staticmethod
    def is_file_exist(s3_client, bucket_name, file_key):
        try:            
            s3_client.get_object(
                Bucket=bucket_name,
                Key=file_key)

            return True
        except ClientError as ex:
            if(ex.response['Error']['Code'] == 'NoSuchKey'):
                return False
            else:
                raise