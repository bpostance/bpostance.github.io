---
layout: post
title:  "Docker Containers: a simple Rest API using Python Flask"
date:   2019-09-05 18:00:00 +0000
comments: true
categories: [tutorial]
tags: [rest-api,docker,containers,gunicorn]
---
You've conducted your research, built your models and packaged everything up in a user friendly application and are ready to share and publish your work within your research community and the wider world. 

There's just one problem. Whilst your models and application run fine on your own development machine, this machine is running on an older university version of Windows Server, you've collected and installed used varous Python and R packages over time, created and set environment variables, configuration files, etc etc. How can someone re-create your environment, run and use all of your hard work? 

Containers solve this problem by packaging up all the software, settings and code used to execute and application within a single and sharable [Docker Container](https://www.docker.com/resources/what-container). In addition, containerised App's enable a wide variety of different operating systems and environments to be used from a single underlying host's operating system and infrastructure.

![png]({{ "assets\images\2019-09-05-docker-containers-fig1.png"}})

This post demonstrates how to setup a simple Docker container to host a Flask API application. The code is avaliable in this [repo](https://github.com/bpostance/training.docker).


### The Flask app
```python
from flask import Flask, jsonify, request

app = Flask(__name__)

# Routes
@app.route('/', methods=['GET'])
def hello_world():
    # flask prints strings as html rendered in browser
	return jsonify(message='Hello World'

if __name__ == "__main__":
	app.run(debug=False)
```

That's all folks<BR>
Thank you for reading.

Ben Postance

1: https://www.docker.com/resources/what-container