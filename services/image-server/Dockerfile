FROM registry.redhat.io/rhel8/python-27

WORKDIR /usr/src/app

COPY . ./

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "-u", "image-server.py"]
