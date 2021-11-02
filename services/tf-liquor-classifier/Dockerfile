FROM nvcr.io/nvidia/tensorflow:21.05-tf2-py3

WORKDIR /usr/src/app

COPY FreeMono.ttf custom-classifier.py class_names.json requirements.txt liquor_model.h5 ./

RUN /usr/bin/python -m pip install --upgrade pip \
 && pip install -r requirements.txt

CMD ["python3", "-u", "custom-classifier.py"]
