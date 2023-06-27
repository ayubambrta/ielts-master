FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y python3 python3-pip -y
RUN apt-get install -y ffmpeg

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3"]

# RUN python3 -m spacy download en_core_web_sm

CMD ["app.py"]