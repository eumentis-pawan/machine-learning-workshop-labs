FROM nvcr.io/nvidia/tensorflow:21.05-tf2-py3

WORKDIR /usr/src/app

COPY requirements.txt risk-assessment.py pneumonia_model.h5 FreeMono.ttf ./

RUN pip install -r requirements.txt

CMD ["python3", "-u", "risk-assessment.py"]
