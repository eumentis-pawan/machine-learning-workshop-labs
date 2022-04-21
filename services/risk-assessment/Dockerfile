FROM registry.redhat.io/rhel8/python-27

WORKDIR /usr/src/app

COPY requirements.txt risk-assessment.py pneumonia_model.h5 FreeMono.ttf ./

RUN pip install -r requirements.txt

CMD ["python3", "-u", "risk-assessment.py"]
