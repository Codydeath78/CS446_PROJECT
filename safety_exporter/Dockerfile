#use an official Node.js runtime as a parent image
FROM node:22-slim
#set the working directory in the container
WORKDIR /app
#copy package.json and package-lock.json to the working directory
COPY package.json package-lock.json* ./
#install dependencies
RUN npm install --production
#copy the rest of the application code to the working directory
COPY index.js ./
#expose the port the app runs on
ENV NODE_ENV=production
#set the port environment variable
ENV PORT=8080
#expose port 8080
CMD ["npm", "start"]