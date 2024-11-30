FROM python:3.9-slim

WORKDIR /app

COPY  requirement.txt requirement.txt

RUN apt-get update && apt-get install -y bash

RUN pip install --no-cache-dir -r requirement.txt


COPY . .

EXPOSE 8000

CMD ["python" , "app.py"]