FROM registry.redhat.io/ubi8/python-38

WORKDIR /usr/src/app

COPY . ./

RUN pip install -r requirements.txt

CMD ["python", "-u", "image-generator.py"]
