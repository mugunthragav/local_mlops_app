# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Install ngrok
RUN wget https://bin.equinox.io/c/111111/ngrok-stable-linux-amd64.zip && \
    unzip ngrok-stable-linux-amd64.zip && \
    mv ngrok /usr/local/bin && \
    rm ngrok-stable-linux-amd64.zip

# Expose the ports for any custom services (if needed)
EXPOSE 8501 5000

# Define the default command to run
CMD ["python", "app.py"]
