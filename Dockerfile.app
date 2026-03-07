FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run feature engineering and train models if not present
# (Removed to prevent Render timeout. Using pre-trained models from GitHub instead.)
# RUN python -m src.features || true
# RUN python -m src.train_yield || true
# RUN python -m src.train_disease || true

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
