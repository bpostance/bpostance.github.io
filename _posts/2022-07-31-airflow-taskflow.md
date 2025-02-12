---
layout: post
title: "Writing Pythonic Airflow DAGs with the TaskFlow API"
date: 2022-07-31 18:00:00 +0000
categories: [blog-post, engineering]
tags: [airflow, data-engineering]
math: false
comments: true
---

This post provides a brief introduction to the Airflow TaskFlow API, demonstrating how to write more Pythonic DAGs. For further details, see the [Airflow TaskFlow Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html).

## Background

I have used Airflow extensively in both professional and personal projects over the years. My setup typically follows the [out-of-the-box configuration for Kubernetes with Helm](https://airflow.apache.org/docs/helm-chart/stable/index.html), utilising the Celery executor, and constructing DAGs primarily composed of tasks executed by Python Virtual Environment operators.

In multi-task DAGs, passing arguments and output variables between task instances can become complex with Airflow XComs, which are embedded within the operator's execution layer. This complexity is further compounded when using Python Virtual Environment operators, as variable communication relies on serialisation (using dill or pickle) before being incorporated into the Airflow context. These challenges are discussed in [this GitHub issue](https://github.com/apache/airflow/issues/20974), and it appears that Airflow is evolving away from the original methods of writing DAGs and tasks.

## Example TaskFlow

The TaskFlow API simplifies the process of passing small variables between tasks using a classic function `return`. Note that the same best practices apply regarding the size and complexity of variables exchanged between tasks.


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