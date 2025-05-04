FROM python:3.10-slim

WORKDIR /Diabetes

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "src/train.py"]