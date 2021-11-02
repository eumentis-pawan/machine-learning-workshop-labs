import io
import logging
import os
import sys
from hashlib import blake2b
from io import BytesIO
import json

import boto3
import numpy as np
import tensorflow as tf
from cloudevents.http import from_http
from flask import Flask, request, Response, jsonify
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import mysql.connector
from flask_cors import CORS

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.info('Num GPUs Available: ' +  str(len(tf.config.list_physical_devices('GPU'))))

##############
## Vars init #
##############
# Object storage
access_key = os.getenv('AWS_ACCESS_KEY_ID', None)
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', None)
service_point = os.getenv('S3_URL_ENDPOINT', 'http://ceph-nano-0/')
s3client = boto3.client('s3','us-east-1', endpoint_url=service_point,
                       aws_access_key_id = access_key,
                       aws_secret_access_key = secret_key,
                        use_ssl = True if 'https' in service_point else False)

# Bucket base name
bucket_base_name = os.getenv('BUCKET_BASE_NAME', 'liquor-images')

# Helper database
db_user = os.getenv('DATABASE_USER', 'liquorlab')
db_password = os.getenv('DATABASE_PASSWORD', 'liquorlab')
db_host = os.getenv('DATABASE_HOST', 'liquorlabdb')
db_db = os.getenv('DATABASE_DB', 'liquorlabdb')

# Inference model version - this gets stored in the DB along with our prediction to analyze between versions
model_version = os.getenv('model_version', '1')

# Load the model
model = tf.keras.models.load_model('./liquor_model.h5')
logging.info('model loaded')

########
# Code #
########
# Main Flask app
app = Flask(__name__)
CORS(app)

#@app.route("/", methods=["POST"])
#def home():
#    # Retrieve the CloudEvent
#    event = from_http(request.headers, request.get_data())
#    
#    # Process the event
#    process_event(event.data)
#
#    return "", 204

# Classificaiton endpoint
@app.route("/", methods=["POST"])
def classify():
  #predictions = dict(run_inference_on_image(request.data))
  #predictions = dict(run_inference_on_image(request.get_data()))
  # Retrieve the CloudEvent
  event = from_http(request.headers, request.get_data())
  
  # Process the event
  predictions = process_inference_event(event.data)

  logging.info(predictions)
  return jsonify(predictions=predictions), 204

def process_inference_event(data):
    """Runs inference on an image against the model"""

    logging.info(data)
    try:
        # Retrieve event info
        extracted_data = extract_data(data)
        # Set simple vars
        bucket_eventName = extracted_data['bucket_eventName']
        bucket_name = extracted_data['bucket_name']
        img_key = extracted_data['bucket_object']
        # Get the Image name by splitting on the last forward-slash in the path
        img_name = img_key.split('/')[-1]
        logging.info('Event: ' + bucket_eventName + ' | Bucket Name: ' + bucket_name + ' | Image: ' + img_key)

        if 's3:ObjectCreated' in bucket_eventName:
            # Load image and make prediction
            logging.info('load_image')
            logging.info('load_image - bucket_name: ' + bucket_name)
            logging.info('load_image - img_key: ' + img_key)
            
            obj = s3client.get_object(Bucket=bucket_name, Key=img_key)
            img = Image.open(io.BytesIO(obj['Body'].read()))
            img = img.convert('RGB')
            img = img.resize((180, 180), Image.NEAREST)
            img_tensor = tf.keras.preprocessing.image.img_to_array(img)                    # (height, width, channels)
            img_tensor = tf.expand_dims(img_tensor, 0)               # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
            #img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

            # Make a prediction
            logging.info('prediction')
            try:
                pred = model.predict(img_tensor)
                logging.info('prediction made')
                score = tf.nn.softmax(pred[0])

                with open('./class_names.json') as json_file:
                    class_names = json.load(json_file)
            
                label=class_names[np.argmax(score)] + ', guess=' + str(round(100 * np.max(score),2)) + '%'

            except Exception as e:
                logging.error(f"Prediction error: {e}")
                raise

            logging.info('label')
            #prediction = {'label':label,'pred':pred[0][0]}
            prediction = {'label':label,'class':class_names[np.argmax(score)],'pred':round(100 * np.max(score),2)}
            logging.info('result=' + prediction['label'])

            # Get original image and print prediction on it
            image_object = s3client.get_object(Bucket=bucket_name,Key=img_key)
            img = Image.open(BytesIO(image_object['Body'].read()))
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('FreeMono.ttf', 50)
            draw.text((-1, 1), prediction['label'], font=font, fill=(255,0,0,255))
            draw.text((1, -1), prediction['label'], font=font, fill=(255,0,0,255))
            draw.text((-1, -1), prediction['label'], font=font, fill=(255,0,0,255))
            draw.text((1, 1), prediction['label'], font=font, fill=(255,0,0,255))
            draw.text((0, 0), prediction['label'], font=font, fill=(255,0,0,255))

            # Save image with "-processed" appended to name
            computed_image_key = os.path.splitext(img_key)[0] + '-processed.' + os.path.splitext(img_key)[-1].strip('.')
            buffer = BytesIO()
            img.save(buffer, get_safe_ext(computed_image_key))
            buffer.seek(0)
            sent_data = s3client.put_object(Bucket=bucket_base_name+'-processed', Key=computed_image_key, Body=buffer)
            if sent_data['ResponseMetadata']['HTTPStatusCode'] != 200:
                raise logging.error('Failed to upload image {} to bucket {}'.format(computed_image_key, bucket_base_name + '-processed'))
            update_images_processed(computed_image_key,model_version,prediction['label'])
            logging.info('Image processed')
            
            return prediction

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise


def process_event(data):
    """Main function to process data received by the container image."""

    logging.info(data)
    try:
        # Retrieve event info
        extracted_data = extract_data(data)
        bucket_eventName = extracted_data['bucket_eventName']
        bucket_name = extracted_data['bucket_name']
        img_key = extracted_data['bucket_object']
        img_name = img_key.split('/')[-1]
        logging.info(bucket_eventName + ' ' + bucket_name + ' ' + img_key)

        if 's3:ObjectCreated' in bucket_eventName:
            # Load image and make prediction
            new_image = load_image(bucket_name,img_key)
            result = prediction(new_image)
            logging.info('result=' + result['label'])

            # Get original image and print prediction on it
            image_object = s3client.get_object(Bucket=bucket_name,Key=img_key)
            img = Image.open(BytesIO(image_object['Body'].read()))
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('FreeMono.ttf', 50)
            draw.text((0, 0), result['label'], (255), font=font)

            # Save image with "-processed" appended to name
            computed_image_key = os.path.splitext(img_key)[0] + '-processed.' + os.path.splitext(img_key)[-1].strip('.')
            buffer = BytesIO()
            img.save(buffer, get_safe_ext(computed_image_key))
            buffer.seek(0)
            sent_data = s3client.put_object(Bucket=bucket_base_name+'-processed', Key=computed_image_key, Body=buffer)
            if sent_data['ResponseMetadata']['HTTPStatusCode'] != 200:
                raise logging.error('Failed to upload image {} to bucket {}'.format(computed_image_key, bucket_base_name + '-processed'))
            update_images_processed(computed_image_key,model_version,result['label'])
            logging.info('Image processed')

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

def extract_data(data):
    logging.info('extract_data')
    record=data['Records'][0]
    bucket_eventName=record['eventName']
    bucket_name=record['s3']['bucket']['name']
    bucket_object=record['s3']['object']['key']
    data_out = {'bucket_eventName':bucket_eventName, 'bucket_name':bucket_name, 'bucket_object':bucket_object}
    logging.info('data extracted')
    return data_out

def load_image(bucket_name, img_path):
    local_path = "/tmp/%s"% (img_path,)
    logging.info('load_image')
    logging.info(bucket_name)
    logging.info(img_path)
    obj = s3client.get_object(Bucket=bucket_name, Key=img_path)
    img = Image.open(io.BytesIO(obj['Body'].read()))
    img = img.convert('RGB')
    img = img.resize((50, 50), Image.NEAREST)
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor

def prediction(new_image):
    logging.info('prediction')
    try:
        pred = model.predict(new_image)
        logging.info('prediction made')
    
        if pred[0][0] > 0.80:
            label='Hendricks, guess=' + str(round(pred[0][0]*100,2)) + '%'
        elif pred[0][0] < 0.60:
            label='Herradura, guess=' + str(round(pred[0][0]*100,2)) + '%'
        else:
            label='Unsure, guess=' + str(round(pred[0][0]*100,2)) + '%'
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise   
    logging.info('label')
    prediction = {'label':label,'pred':pred[0][0]}
    return prediction

def get_safe_ext(key):
    ext = os.path.splitext(key)[-1].strip('.').upper()
    if ext in ['JPG', 'JPEG']:
        return 'JPEG' 
    elif ext in ['PNG']:
        return 'PNG' 
    else:
        logging.error('Extension is invalid')

def update_images_processed(image_name,model_version,label):
    try:
        cnx = mysql.connector.connect(user=db_user, password=db_password,
                                      host=db_host,
                                      database=db_db)
        cursor = cnx.cursor()
        query = 'INSERT INTO images_processed(time,name,model,label) SELECT CURRENT_TIMESTAMP(), "' + image_name + '","' + model_version + '","' + label.split(',')[0] + '";'
        cursor.execute(query)
        cnx.commit()
        cursor.close()
        cnx.close()

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise


# Launch Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0')
