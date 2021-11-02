import logging
import os
import sys
from string import Template

import flask
from flask import Flask
from flask_cors import CORS
import mysql.connector

import boto3

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

##############
## Vars init #
##############
# Helper database
db_user = os.getenv('DATABASE_USER', 'liquorlab')
db_password = os.getenv('DATABASE_PASSWORD', 'liquorlab')
db_host = os.getenv('DATABASE_HOST', 'liquorlabdb')
db_db = os.getenv('DATABASE_DB', 'liquorlabdb')

# S3 Endpoint
access_key = os.getenv('AWS_ACCESS_KEY_ID', None)
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', None)
service_point = os.getenv('S3_URL_ENDPOINT', 'http://ceph-nano-0/')
s3client = boto3.client('s3', 'us-east-1', endpoint_url=service_point,
                        aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key,
                        use_ssl=True if 'https' in service_point else False)

# Bucket base name
bucket_base_name = os.getenv('BUCKET_BASE_NAME', 'liquor-images')

########
# Code #
########
def get_last_image(bucket_name):
    """Retrieves the last uploaded image in a bucket, according to helper database timestamp."""

    # Correspondance table between bucket names and table names. Yeah, I know...
    bucket_table = {bucket_base_name:'images_uploaded',bucket_base_name+'-processed':'images_processed',bucket_base_name+'-anonymized':'images_anonymized',}
    try:
        cnx = mysql.connector.connect(user=db_user, password=db_password,
                                      host=db_host,
                                      database=db_db)
        cursor = cnx.cursor()
        query = 'SELECT name FROM ' + bucket_table[bucket_name] + ' ORDER BY time DESC LIMIT 1;'
        cursor.execute(query)
        data = cursor.fetchone()
        if data is not None:
            result = data[0]
        else:
            result = ""
        cursor.close()
        cnx.close()

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

    return result

# Response templates 
HOME_TEMPLATE = Template("""
<!DOCTYPE html>
<html>
<body>

<h1>Liquor Images</h1>
<h2>Last Uploaded Image</h2>
${last_uploaded}
<h2>Last Proccessed Image</h2>
${last_processed}
</body>
</html>
""")

LOCATION_TEMPLATE_SMALL = Template("""
    <img src="/download_image/${bucket_name}/${image_name}" style="width:260px;"></img>""")

LOCATION_TEMPLATE_BIG = Template("""
    <img src="/download_image/${bucket_name}/${image_name}" style="width:575px;"></img>""")

# Main Flask app
app = Flask(__name__)
CORS(app)

# Test route
@app.route('/')
def homepage():
    html = HOME_TEMPLATE.substitute(last_uploaded=last_image_small(bucket_base_name),
                                    last_processed=last_image_small(bucket_base_name + '-processed'))
    
    return html

# Answers with last image from <bucketname>, formatted as small
@app.route('/last_image_small/<bucket_name>')
def last_image_small(bucket_name):
    image_name = get_last_image(bucket_name)
    if image_name != "":   
        html = LOCATION_TEMPLATE_SMALL.substitute(bucket_name=bucket_name, image_name=image_name)
    else:
        html = '<h2 style="font-family: Roboto,Helvetica Neue,Arial,sans-serif;text-align: center; color: white;font-size: 15px;font-weight: 400;">No image to show</h2>'
    return html

# Answers with last image from <bucketname>, formatted as big
@app.route('/last_image_big/<bucket_name>')
def last_image_big(bucket_name):
    image_name = get_last_image(bucket_name)   
    if image_name != "":   
        html = LOCATION_TEMPLATE_BIG.substitute(bucket_name=bucket_name, image_name=image_name)
    else:
        html = '<h2 style="font-family: Roboto,Helvetica Neue,Arial,sans-serif;text-align: center; color: white;font-size: 15px;font-weight: 400;">No image to show</h2>'
    return html

# Download an image from an S3 bucket and send it back to the browser.
@app.route('/download_image/<bucket_name>/<image_name>')
def download_image(bucket_name, image_name):
    logging.info("Request: bucket_name, %s; image_name, %s"% (bucket_name, image_name))
    obj = s3client.get_object(Bucket=bucket_name, Key=image_name)

    response = flask.Response(mimetype='image/jpeg')
    response.data = obj['Body'].read()
    return response

# Launch Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0')

