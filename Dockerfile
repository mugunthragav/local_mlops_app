# Use Python base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the project files to the working directory
COPY . /app

# Install the required Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose ports for Streamlit (8501) and MLflow (5000)
EXPOSE 8501
EXPOSE 5000

# Default command (entry point) for running the container
CMD ["bash", "entrypoint.sh"]
