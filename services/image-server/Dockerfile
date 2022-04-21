FROM registry.redhat.io/ubi8/python-38

WORKDIR /usr/src/app

COPY . ./

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "-u", "image-server.py"]
