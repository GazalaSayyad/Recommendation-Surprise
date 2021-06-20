
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY ./app /app
WORKDIR /app
COPY . /requirements.txt/requirements.txt
RUN pip install -r requirements.txt


