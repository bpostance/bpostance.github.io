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


![png]({{ "assets/images/2019-09-05-docker-containers-fig1.png" }})

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

### Gunicorn
https://vsupalov.com/what-is-gunicorn/


That's all folks<BR>
Thank you for reading.

Ben Postance

1: https://www.docker.com/resources/what-container