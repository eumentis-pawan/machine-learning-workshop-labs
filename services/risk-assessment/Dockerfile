FROM registry.redhat.io/ubi8/python-38

WORKDIR /usr/src/app

COPY requirements.txt risk-assessment.py pneumonia_model.h5 FreeMono.ttf ./

RUN pip install -r requirements.txt

CMD ["python3", "-u", "risk-assessment.py"]
