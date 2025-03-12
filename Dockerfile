# Dockerfile for building a Python application container
# This Dockerfile sets up a container environment for running a Python application.
# It installs the necessary dependencies from requirements.txt, sets up environment variables,
# and runs the main Python script app.py when the container starts.
# Prerequisites: Docker installed on the host system


# Use the latest Python image as the base. This image will provide the necessary Python runtime environment
FROM python:3.8.12


# Copy the necessary files into the working directory (/app) in the container
# This includes all files from the current directory (where the Dockerfile is located) into the /app directory within the container
COPY . /app


# Set the working directory to /app. This directory will be the default location for any subsequent commands executed within the container
WORKDIR /app


# Define mandatory environment variables without default values
# These variables are essential for the proper functioning of the application, and users must provide values for them during container runtime

# The port on which the application will listen for incoming connections
ENV PORT=8080
# The basepath environment variable
ENV BASE_PATH='/explanation'

# The URL of the message broker to connect to
ENV BROKER_URL=https://orion.tema.digital-enabler.eng.it
# The ID of the topic to subscribe to or publish messages to
ENV BROKER_ENTITY_ID=topic123
# The type of the entity (topic or queue) to interact with
ENV BROKER_TYPE_ID=type123
# The ID of the subscription to use for receiving messages
ENV BROKER_SUBSCRIPTION_ID=subscription123
# The URL to which callback notifications will be sent
ENV CALLBACK_URL=http://callback.example.com


# Define environment variables with default values
# These variables are optional, and if not provided, the default values will be used

# Whether debugging mode is enabled. Default is set to False
ENV DEBUG=False        
# The processing unit to be used (cpu or gpu). We set it to gpu
ENV PROCESSING_UNIT=gpu 


# Install the dependencies specified in requirements.txt
# This command installs all Python packages listed in the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Dependencies for CV2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# Run the application when the container starts
# This command specifies the default command to execute when the container is launched
# It runs the Python script named "app.py", which is the main entry point for the application
CMD ["python", "app.py"]