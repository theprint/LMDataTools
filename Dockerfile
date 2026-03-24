# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install system dependencies that might be needed for networking and other libraries
# ca-certificates is important for SSL/TLS verification
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
# This includes webapp.py, the tool scripts (datapersona.py, etc.),
# the datacore directory, and the webapp directory with static files.
COPY . .

# Set the timezone for the container to match your local timezone (PST/PDT)
# This ensures that timestamps generated inside the container are correct.
ENV TZ="America/Los_Angeles"

# Set up a directory for model caching and define environment variables
# for common model providers to use this directory.
ENV HF_HOME=/app/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers

# Create the directory to avoid any potential write permission errors on startup
RUN mkdir -p /app/models

# Make port 8910 available to the world outside this container
EXPOSE 8910

# Define the command to run your app using uvicorn
# The host 0.0.0.0 makes the server accessible from outside the container.
CMD ["uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "8910"]