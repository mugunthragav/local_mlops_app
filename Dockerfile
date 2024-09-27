# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install ngrok
RUN apt-get update && apt-get install -y wget unzip && \
    wget https://bin.equinox.io/c/111111/ngrok-stable-linux-amd64.zip && \
    unzip ngrok-stable-linux-amd64.zip && \
    mv ngrok /usr/local/bin && \
    rm ngrok-stable-linux-amd64.zip

# Expose the ports for MLflow and Streamlit
EXPOSE 5000 8501

# Run MLflow and Streamlit in the background
CMD ["sh", "-c", "mlflow ui --host 0.0.0.0 --port 5000 & streamlit run app.py --server.port 8501 --server.headless true"]
