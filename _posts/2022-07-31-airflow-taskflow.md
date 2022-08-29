---
layout: post
title:  "Writing Pythonic Airflow DAGs with TaskFlow API "
date:   2022-07-31 18:00:00 +0000
categories: [blog-post,engineering]
tags: [airflow,data-engineering]
math: false
comments: true
---

This post is a brief introduction to Airflow TaskFlow to write more Pythonic DAGs. The [Airflow TaskFlow Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html) 


## Background
I've been using Airflow at work and personal projects for a number of years. This has generally followed the [OOTB configuration for kubernetes with helm](https://airflow.apache.org/docs/helm-chart/stable/index.html), using the Celery executor, and writing DAGs that primarily consist of tasks executed by PythonVirtualEnv operators.

For multi-task DAGs, passing arguments and output variables between task instances can become quite complex using Airflow XCom. XComs are hidden in the Airflow execution layer inside the operator. With PythonVirtualEnv operators this is further complicated given how Airflow handles variable communication between tasks. This involves serialisation of variables using dill or pickle, and adding these to Airflow context. These are described in this [Github Issue](https://github.com/apache/airflow/issues/20974) and it seems Airflow are moving away from the "original" way of writing DAGs and Tasks. 

## Example TaskFlow

Using taskflow to pass small variables using classic function `return`. The same rules still apply with respect to ony passing small variables between tasks.

```python
from datetime import datetime
from airflow.decorators import dag, task
from airflow.operators.python_operator import PythonVirtualenvOperator
from typing import Dict

@dag(   
    start_date=datetime.now(),
    schedule_interval='@once',
    catchup=False,
    tags=['helloworld'],
    description='example Task Flow API',
    is_paused_upon_creation=True
    )
def example_taskflow():
    """
    This is a simple TaskFlow pipeline 
    """
    @task()
    def task1() -> Dict[int, str]:
        """
        Simple task to create and return a dict of customers
        """
        customers = {1:"Chandler",2:"Rachel",3:"Monica"}

        return customers
    
    @task.virtualenv(requirements=['pandas'])
    def task2(customers: dict):
        """
        This task runs in a python virtual env operator:
        - We install pandas as a requirement
        - Add a customer
        - And create a table of customers
        """
        import os
        import pandas as pd

        new_customers = {4:"Phoebe",5:"Ross"}
        customers.update(new_customers)

        df = pd.DataFrame.from_dict(customers,orient='index',columns=['first_name'])
        
        # output larger objects to datastore
        #df.to_sql()
        print(df)

    customers = task1()
    task2(customers)

my_dag = example_taskflow()
```

And this example shows how to blend TaskFlow format tasks with typical Airflow task:operator syntax.

```python
from datetime import datetime
from airflow.decorators import dag, task
from airflow.operators.python_operator import PythonVirtualenvOperator
from typing import Dict

@dag(   
    start_date=datetime.now(),
    schedule_interval='@once',
    catchup=False,
    tags=['helloworld'],
    description='example Task Flow API',
    is_paused_upon_creation=True
    )
def example_taskflow():
    """
    This is a simple TaskFlow pipeline 
    """
    @task()
    def task1() -> Dict[int, str]:
        """
        Simple task to create and return a dict of customers
        """
        customers = {1:"Chandler",2:"Rachel",3:"Monica"}

        return customers
    
    @task.virtualenv(requirements=['pandas'])
    def task2(customers: dict):
        """
        This task runs in a python virtual env operator:
        - We install pandas as a requirement
        - Add a customer
        - And create a table of customers
        """
        import os
        import pandas as pd

        new_customers = {4:"Phoebe",5:"Ross"}
        customers.update(new_customers)

        df = pd.DataFrame.from_dict(customers,orient='index',columns=['first_name'])
        #df.to_sql()
        print(df)

    def task():
        other_person = {6:'Gunther'}
        return  other_person

    task3 = PythonVirtualenvOperator(
        task_id='task_3', 
        python_callable=task,
        )

    task2(task1()) >> task3

my_dag = example_taskflow()
```


## References
- https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html
- https://www.astronomer.io/events/recaps/taskflow-api-in-airflow-2-0/