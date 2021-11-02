import logging
import os
import random
import sys
from time import sleep

import boto3
import mysql.connector
import requests
from botocore import UNSIGNED
from botocore.client import Config

from kafka import KafkaProducer
import json


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

##############
## Vars init #
##############
# Object storage
access_key = os.getenv('AWS_ACCESS_KEY_ID', None)
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', None)
service_point = os.getenv('S3_URL_ENDPOINT', 'http://ceph-nano-0/')
s3client = boto3.client('s3', 'us-east-1', endpoint_url=service_point,
                        aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key,
                        use_ssl=True if 'https' in service_point else False)

s3sourceclient = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Buckets
bucket_source = os.getenv('BUCKET_SOURCE', 'https://polyglot-academy-pub.nyc3.digitaloceanspaces.com/liquor-ml')
bucket_source_name = bucket_source.split('/')[-1]
bucket_destination = os.getenv('BUCKET_BASE_NAME', 'liquor-images')

# Helper database
db_user = os.getenv('DATABASE_USER', 'liquorlab')
db_password = os.getenv('DATABASE_PASSWORD', 'liquorlab')
db_host = os.getenv('DATABASE_HOST', 'liquorlabdb')
db_db = os.getenv('DATABASE_DB', 'liquorlabdb')

# Delay between images
seconds_wait = float(os.getenv('SECONDS_WAIT', 2))

# Kakfa producer
producer = KafkaProducer(bootstrap_servers='my-cluster-kafka-bootstrap:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

########
# Code #
########
def copy_file(source, image_key, destination, image_name):
    """Copies an object from a URL source to a destination bucket.""" 

    image_url = source + '/' + image_key
    req_for_file = requests.get(image_url, stream=True)

    # Init File-like object (to be used by upload_fileobj method)
    file_object_from_req = req_for_file.raw

    s3client.upload_fileobj(file_object_from_req,destination,image_name)

def send_event(destination, image_name):
    """Sends an SMS event to the Kafka broker"""
    
    record = {'eventName': 's3:ObjectCreated',
              's3': {'bucket': {'name': destination},
                     'object': {'key': image_name}
                    }
             }
    logging.info("sending %s"% record)
    
    producer.send('liquor-images', {'Records': [record]})
    
def update_images_uploaded(image_name):
    """Inserts image name and timestamp into the helper database."""

    try:
        cnx = mysql.connector.connect(user=db_user, password=db_password,
                                      host=db_host,
                                      database=db_db)
        cursor = cnx.cursor()
        query = 'INSERT INTO images_uploaded(time,name) SELECT CURRENT_TIMESTAMP(),"' + image_name + '";'
        cursor.execute(query)
        cnx.commit()
        cursor.close()
        cnx.close()

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

# Populate source images lists
hendricks_images=[]
for image in s3sourceclient.list_objects(Bucket=bucket_source_name,Prefix='HENDRICKS/val/')['Contents']:
    hendricks_images.append(image['Key'])
herradura_images=[]
for image in s3sourceclient.list_objects(Bucket=bucket_source_name,Prefix='HERRADURA/val/')['Contents']:
    herradura_images.append(image['Key'])

# Main loop
while seconds_wait != 0: #This allows the container to keep running but not send any image if parameter is set to 0
    logging.info("copy image")
    # Quick randomness
    rand_type = random.randint(1,10)
    if rand_type <= 6: # 60% of time, choose a Hendricks image
        image_key = hendricks_images[random.randint(0,len(hendricks_images)-1)]
    else:
        image_key = herradura_images[random.randint(0,len(herradura_images)-1)]
    image_name = image_key.split('/')[-1]
    # Copy file from source bucket to target bucket
    copy_file(bucket_source,image_key,bucket_destination,image_name)
    # Send the event to the Kafka Topic
    send_event(bucket_destination,image_name)
    # Add the image to the images_uploaded Helper DB
    update_images_uploaded(image_name)
    # Sleep and come back and do it again
    sleep(seconds_wait)

# Dirty hack to keep container running even when no images are to be copied
os.system("tail -f /dev/null")
