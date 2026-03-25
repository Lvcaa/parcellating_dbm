FROM python:3.11-slim

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default: run the script
CMD ["python", "fake_matrix_irbio.py"]
