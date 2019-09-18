---
layout: post
title:  "Docker Containers: a simple Rest API using Python Flask"
date:   2019-09-05 18:00:00 +0000
comments: true
categories: [tutorial]
tags: [rest-api,docker,containers,gunicorn]
---
You've conducted your research, analysed the data, built models, and packaged everything up in a user friendly application that is ready to be shared and published within your research community and beyond. 

There's just one problem. Whilst your models and application run fine on your own development machine, perhaps this machine is running on an older University Windows Server, or you've collected and installed various packages and code libraries over time, set environment variables, configurations, etc etc. How can someone re-create your environment to run and use all of your hard work? 

Containers solve this problem by packaging up all of the software, settings and code used to execute and application within a single and sharable [Docker Container](https://www.docker.com/resources/what-container). In addition, containerization enables a wide variety of different operating systems and environments to be used from a single underlying host's operating system and infrastructure.


![png]({{ "/assets/images/2019-09-05-docker-containers-fig1.png" }})

This post demonstrates how to setup a simple Docker container to run an API using Python Flask. The code is avaliable in this [repo](https://github.com/bpostance/training.docker).

## Install Docker
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
├── my_app
│   ├── start.ps1      		<- Windows powershell to build and run our docker
│   ├── Dockerfile     		<- Instructions to build our container
│   ├── environment.yml		<- Conda environment.yml
│   ├── serve.sh			<- bash script to run an application server
│   └── raw            		
│   	└── flask-api.py	<- the flask app
```
There are few new files and terms in here that are explained below. 

## The Flask App
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


### Application Server
Our app runs fine locally for debugging but this wont fly in production. 
Web applications generally require:
 - **A web server (like nginx).** The web server accepts requests, takes care of general domain logic and takes care of handling https connections. 
 - **A WSGI application server (like Gunicorn).**  The application server handles the requests which are meant to arrive at the application itself.
 - **The application.**

Here we are only going to worry about our Application Server and our App. This post isn't intended to explain web servers and requests, however see [this SO answer](https://serverfault.com/a/331263) if you're interested and want to learn more.
For the Application Server we will use [Gunicorn](https://gunicorn.org/) is more robust than Flasks internal debugging server we used above, it helps to:
 - host files
 - handle conncetions
 - manage server errors and issues
 - improves scalability

Here's the code to run our app in gunicorn. Nice and simple we, change directory to /app, bind a server socket, and assign our "flask-api" as the generic "app".

```
# serve.sh
#!/bin/bash
# run with gunicorn (http://docs.gunicorn.org/en/stable/run.html#gunicorn)
exec gunicorn --chdir app  -b :5000 flask-api:app
```


That's all folks<BR>
Thank you for reading.

Ben Postance

1: https://www.docker.com/resources/what-container