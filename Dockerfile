FROM python:3.12-slim

WORKDIR /app

# Dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Code
COPY . .

# Streamlit runs at 8501
EXPOSE 8501

# Streamlit needs in Container 0.0.0.0
CMD ["streamlit", "run", "app_v2.py", "--server.address=0.0.0.0", "--server.port=8501"]
