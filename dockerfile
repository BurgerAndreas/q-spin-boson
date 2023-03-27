FROM python:3.10-slim
# FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /app

# Already installed in python:3.10-slim
# RUN apt-get update && apt-get install -y 
# RUN pip install --upgrade pip

# generated from conda with
# pip list --format=freeze > requirements.txt
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the source code and data into the container
COPY ./q-spin-boson/ .

# bind mount:
# Directory on the host machine that is mounted into the container at runtime
# Bind Mount = file/directory on your local machine available inside your container. 
# Any changes you make to this file/directory from outside the container 
# will be reflected inside the docker container and vice-versa
# docker run -v <source local>:<target container>

# volumes:
# can be used to share data between multiple containers
# volumes are stored in a part of the host filesystem which is managed by Docker
# volumes are independent of the container
# one can access volumes from inside the container using the terminal
# VOLUME $(HOME)/Coding/q-spin-boson/q-spin-boson/data

# Set environment variables for paths
# ENV DIR_MODEL_SAVE=$(pwd)/q-spin-boson/data
# DIR_MODEL_SAVE = os.environ["DIR_MODEL_SAVE"] # in main.py
# ENV DIR_DATA="data"

# RUN python3 main.py
CMD ["python","-u","main.py"]

# usage
# start docker daemon / desktop app
# $ docker build -t myimage -f dockerfile .
# $ docker images
# # $ docker run -it --name mycontainer myimage
# $ docker run -it -v $(pwd)/project/data:/app/data --name mycontainer myimage
# $ docker ps -a
# $ docker rm -f mycontainer