FROM quay.io/bitnami/python:3.7

WORKDIR /usr/src/app

COPY . ./

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "-u", "image-server.py"]
