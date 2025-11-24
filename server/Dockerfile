#use an official Python runtime as a parent image
FROM python:3.11-slim
#set the working directory in the container
WORKDIR /app
#copy the current directory contents into the container at /app
COPY requirements.txt .
#install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
#copy the rest of the application code to the container
COPY app.py .
#set environment variable to ensure output is not buffered
ENV PYTHONUNBUFFERED=1
#expose port 8080 for the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]