# Use a lightweight Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && pip install --upgrade pip \
        && pip install -r requirements.txt \
        && apt-get purge -y --auto-remove build-essential \
        && rm -rf /var/lib/apt/lists/*

# Copy the rest of the project
COPY . .

# Optional: create non-root user for security
RUN useradd -m appuser
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit in headless mode accessible from any host
CMD ["streamlit", "run", "dashboard/app.py", "--server.headless=true", "--server.address=0.0.0.0"]
