FROM nvcr.io/nvidia/tensorflow:21.05-tf2-py3

WORKDIR /usr/src/app

COPY requirements.txt liquor-classifier.py liquor_model.h5 FreeMono.ttf ./

RUN /usr/bin/python -m pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["python3", "-u", "liquor-classifier.py"]
