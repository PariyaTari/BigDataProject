FROM python:3.10-bullseye

WORKDIR /src

COPY requirements.txt .

RUN pip install -U pip
RUN pip install -r requirements.txt

COPY . /src

COPY cifar10_model.h5 /src/cifar10_model.h5
RUN chmod 644 /src/cifar10_model.h5

EXPOSE 8000

CMD ["python","main.py"]


