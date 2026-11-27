# Hugging Face Spaces (Docker SDK) / any container host. Hosted mode shows the server-side-processing warning.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Uncomment once you have chosen a model and checked the host's memory limit:
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-ml.txt
COPY . .
ENV LBRIEF_HOSTED=1
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
