FROM python:3.11-slim

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default: run the benchmark script from the reorganized scripts folder
CMD ["python", "scripts/benchmarks/fake_matrix_stress_test.py"]
