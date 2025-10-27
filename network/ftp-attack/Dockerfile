FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ids.py .
CMD ["python", "ids.py", "/flows/ftp_flows.csv"]
