# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project into the container at /app
COPY . /app

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports your app will run on
EXPOSE 8100 9100 3000

# Set environment variables (if you have any)
# You can also pass environment variables through the Docker CLI
ENV NODE_PORT=9100
ENV COORDINATOR_PORT=8100
ENV DASHBOARD_PORT=3000

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
