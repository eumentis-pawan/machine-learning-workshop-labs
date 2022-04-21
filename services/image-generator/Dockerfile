FROM registry.redhat.io/rhel8/python-27

WORKDIR /usr/src/app

COPY . ./

RUN pip install -r requirements.txt

CMD ["python", "-u", "image-generator.py"]
