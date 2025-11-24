#use an official Python runtime as a parent image
FROM python:3.11-slim
#set the working directory in the container
WORKDIR /app
#install any needed packages specified in requirements.txt
COPY requirements.txt .
#install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#this will copy code
COPY . .

#this will expose port
ENV PORT=8080
#expose the port the app runs on
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]