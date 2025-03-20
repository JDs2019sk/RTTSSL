FROM python:3.10-slim

# Install system dependencies for OpenCV and GUI
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for persistence
RUN mkdir -p models datasets logs recordings faces

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV QT_X11_NO_MITSHM=1

# Run the main application
CMD ["python", "main.py"]