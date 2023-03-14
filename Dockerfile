FROM python:3.11-slim

WORKDIR /code

ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r /requirements.txt

ADD src ./src

ADD * /code/

CMD python3 -u main.py