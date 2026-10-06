# CPU-only image for hosting the demo (Hugging Face Spaces "Docker" SDK,
# Render, Fly.io, Cloud Run, or any container host).
#
#   docker build -t htr-demo .
#   docker run -p 8501:8501 htr-demo     # then open http://localhost:8501
#
# Requires models/final_model.pt (create it with
# `python src/export_final_model.py --model baseline`) to be present when
# you build.
FROM python:3.12-slim

WORKDIR /app

COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY .streamlit/ .streamlit/

ENV HTR_HOSTED=1
EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
