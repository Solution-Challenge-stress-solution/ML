# Python 이미지를 기반으로 합니다.
FROM python:3.9

# 작업 디렉토리를 설정합니다.
WORKDIR /app

# 필요한 패키지들을 설치합니다.
COPY requirements.txt .
RUN pip install -r requirements.txt

# protobuf 컴파일러와 관련 라이브러리를 설치합니다.
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

# 소스 코드를 이미지에 복사합니다.
COPY . .

# 모델 파일을 이미지에 복사합니다.
COPY ./SentimentAnalysis/saved_model.pb /app/model/saved_model.pb
COPY ./SentimentAnalysis /app/model

# protobuf 파일을 컴파일합니다.
RUN mkdir -p proto && protoc -I./proto --python_out=./proto ./proto/*.proto

# PYTHONPATH 환경 변수를 설정합니다.
ENV PYTHONPATH=/app

ENV DOCKER_ENV=True

# 서버를 실행합니다.
CMD ["python", "server.py"]


