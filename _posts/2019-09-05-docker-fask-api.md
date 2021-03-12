---
layout: post
title:  "How to containerize a simple Rest API using Python Flask"
date:   2019-09-05 18:00:00 +0000
categories: [blog-post,engineering]
tags: [rest-api,docker,containers,gunicorn,anaconda]
math: true
comments: true
---
You've conducted your research, analysed the data, built models, and packaged everything up in a user friendly application that is ready to be shared and published within your research community and beyond. 

There's just one problem. Whilst your models and application run fine on your own development machine, perhaps this machine is running on an older University Windows Server, or you've collected and installed various packages and code libraries over time, set environment variables, configurations, etc etc. How can someone re-create your environment to run and use all of your hard work? 

Containers solve this problem by packaging up all of the software, settings and code used to execute and application within a single and sharable [Docker Container](https://www.docker.com/resources/what-container). In addition, containerization enables a wide variety of different operating systems and environments to be used from a single underlying host's operating system and infrastructure.


![png]({{ "/assets/images/2019-09-05-docker-containers-fig1.png" }})

This post demonstrates how to setup a simple Docker container to run an API using Python Flask. The code is avaliable in this [repo](https://github.com/bpostance/training.docker-flask-api).

## Install Docker
**These instructions are for Windows 10 OS.**
**See [these instructions for Ubunut/Linux Mint 19 installation](https://computingforgeeks.com/install-docker-and-docker-compose-on-linux-mint-19/).**
Create an account, download and install Docker Desktop for your operating system following the [official channel](https://www.docker.com/products/docker-desktop).
Once installed you check some basics using the following commands. See the git repo readme and official docs for more. 

```
# print docker version info
$ docker -v  

# list images
$ docker images

# list containers
$ docker ps -a
```

## Create a project directory
Our app has a simple structure. 
```
├── conda-flask-api
│   ├── start.ps1      		<- Windows powershell to build and run our docker
│   ├── Dockerfile     		<- Instructions to build our container
│   ├── environment.yml		<- Conda environment.yml
│   ├── serve.sh			<- bash script to run an application server
│   └── raw            		
│   	└── flask-api.py	<- the flask app
```
There are few new files and terms in here that are explained below. 

## flask-api.py
Flask is a lightweight framework for building web applications. If you're new check out the [quickstart](https://flask.palletsprojects.com/en/1.1.x/quickstart/#quickstart).

Lets define some requirements for our app:
1. at root, print some information about the API and show a valid sample request.
2. return the square of the "value" supplied by the user.

```
# flask-app.py
from flask import Flask, request
import json

# create a Flask instance
app = Flask(__name__)

# a simple description of the API written in html.
# Flask can print and return raw text to the browser. 
# This enables html, json, etc. 

description =   """
                <!DOCTYPE html>
                <head>
                <title>API Landing</title>
                </head>
                <body>  
                    <h3>A simple API using Flask</h3>
                    <a href="http://localhost:5000/api?value=2">sample request</a>
                </body>
                """
				
# Routes refer to url'
# our root url '/' will show our html description
@app.route('/', methods=['GET'])
def hello_world():
    # return a html format string that is rendered in the browser
	return description

# our '/api' url
# requires user integer argument: value
# returns error message if wrong arguments are passed.
@app.route('/api', methods=['GET'])
def square():
    if not all(k in request.args for k in (["value"])):
        # we can also print dynamically 
        # using python f strings and with 
        # html elements such as line breaks (<br>)
        error_message =     f"\
                            Required paremeters : 'value'<br>\
                            Supplied paremeters : {[k for k in request.args]}\
                            "
        return error_message
    else:
        # assign and cast variable to int
        value = int(request.args['value'])
        # or use the built in get method and assign a type
        # http://werkzeug.palletsprojects.com/en/0.15.x/datastructures/#werkzeug.datastructures.MultiDict.get
        value = request.args.get('value', type=int)
        return json.dumps({"Value Squared" : value**2})

if __name__ == "__main__":
	# for debugging locally
	# app.run(debug=True, host='0.0.0.0',port=5000)
	
	# for production
	app.run(host='0.0.0.0', port=5000)
```

You can run your app directly in several ways:
1. directly in python. allows debugging. 

```
$python flask-api.py
 * Serving Flask app "flask-api" (lazy loading)```
 
2. or using Flask's builtin server.

```
$ export FLASK_APP=flask-api.py
$ flask run
 * Running on http://0.0.0.0:5000/
```

## serve.sh
Our app runs fine locally for debugging but this wont fly in production. 
Web applications generally require:
 - **A web server (like nginx).** The web server accepts requests, takes care of general domain logic and takes care of handling https connections. 
 - **A WSGI application server (like Gunicorn).**  The application server handles the requests which are meant to arrive at the application itself.
 - **The application.**

Here we are only going to worry about our Application Server and our App. This post isn't intended to explain web servers and requests, however see [this SO answer](https://serverfault.com/a/331263) if you're interested and want to learn more.

For the Application Server we will use [Gunicorn](https://gunicorn.org/). It is more robust than Flasks internal debugging server we used above in that it:
 - host files
 - handles conncetions
 - manages server errors and issues
 - improves scalability

Here's the script that run's our flask-api app in gunicorn. 
The arguments: change source directory to '/app', bind a server socket '5000', and assign our "flask-api" as the generic "app".

```
# serve.sh
#!/bin/bash
# run with gunicorn (http://docs.gunicorn.org/en/stable/run.html#gunicorn)
exec gunicorn --chdir app  -b :5000 flask-api:app
```

## environment.yml
Nothing fancy here i'm using [Conda](https://docs.conda.io/en/latest/) but you could also use pip virtualenv.
```
name: base
channels:
- defaults
dependencies:
- python=3.7
- flask
- gunicorn
```

## Dockerfile
This is the file that we pass to Docker and lists the instructions used to build and execute our container.
In a similar fashion to Git, Docker Hub hosts official and community developed Docker images for popular operating systems and deployments.
Here we use a [debian/miniconda environment from contiuumio](https://hub.docker.com/r/continuumio/miniconda/). You can even check out the [Dockerfile](https://hub.docker.com/r/continuumio/miniconda/dockerfile) to see how this image itself is built.

```
# pull the image from docker hub
FROM continuumio/miniconda3:latest

# adds metadata to an image
LABEL MAINTAINER="Ben Postance"
LABEL GitHub="https://github.com/bpostance/training.docker"
LABEL version="0.0"
LABEL description="A Docker container to serve a simple Python Flask API"

## Override the default shell (not supported on older docker, prepend /bin/bash -c )
SHELL ["/bin/bash", "-c"]

# Set WORKDIR - the working directory for any RUN, CMD, ENTRYPOINT, COPY and ADD instructions that follow it in the Dockerfile
WORKDIR /home/flask-api

# COPY - copies files or directories from <src> and adds them to the filesystem of the container at the path <dest>.
COPY environment.yml ./

# ADD - "adds" directories and their contents to the container
ADD app ./app

# chmod - modifies the boot.sh file so it can be recognized as an executable file.
COPY serve.sh ./
RUN chmod +x serve.sh

# conda set-config and create environment based on .yml
# chain seperate multi-line commands using '&& \'
RUN conda env update -f environment.yml

# set env variables
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate" >> ~/.bashrc

# EXPOSE - informs Docker that the container listens on the specified network ports at runtime
EXPOSE 5000

# ENTRYPOINT - allows you to configure a container that will run as an executable.
ENTRYPOINT ["./serve.sh"]
```

## Build & Run docker 
This last file brings everything together in Docker. 

To run this file you will need to be within the project root /conda-flask-api. 
First, docker build and tag your image. The standard format is "type/name:version".
The "." references the "./Dockerfile".

When you run Docker build docker will print step by step information and raise any issues in the terminal. When getting started it can be helpful to add additional prints to see exactly what docker is doing e.g. "RUN pwd", "RUN ls" etc.

```
# docker build
docker build -t demo/flask-api:0.0 .
```

and to run the container:

```
# docker run
# --name assign name for ease of reference
# -d to run in detached mode
# -p to bind container:local ports
# tag of the container to run
docker run --name demo-flask-api -d -p 5000:5000 demo/flask-api:0.0
```

Now when you inspect running dockers you will see your container.
```
$ docker ps                                                                 

CONTAINER ID        IMAGE                COMMAND             CREATED             STATUS              PORTS                    NAMES
6d4ea8c141df        demo/flask-api:0.1   "./serve.sh"        52 seconds ago      Up 51 seconds       0.0.0.0:5000->5000/tcp   demo-flask-api    
```

And visit [http://localhost:5000/](http://localhost:5000/) or [http://localhost:5000/api?value=2](http://localhost:5000/api?value=2) to visit your api.


That's all folks<BR>
I am writing a follow up post that explains how to use [docker-compose](https://docs.docker.com/compose/) to create multi-container applications.

Thank you for reading.

Ben Postance

1: https://www.docker.com/resources/what-container
