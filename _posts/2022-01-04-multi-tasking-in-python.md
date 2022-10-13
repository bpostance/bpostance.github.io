---
layout: post
title:  "Multi-tasking in Python"
date:   2022-01-04 18:00:00 +0000
categories: [blog-post,engineering]
tags: [parallel-processing,multi-threading]
math: false
comments: true
---

In computing systems, multitasking is the concurrent execution of tasks and processes. We divide up our resources to work on more than one task at the same time.

In data engineering, analytics, and data science we are often faced with scenarios where it is necessary to optimize the speed of execution. In some cases these problems are handled by the data processing and modelling techniques themselves, such as map-reduce or distributed data processing on Apache Spark. But in other cases the time of execution may be driven simply by the volume of tasks. For instance, obtaining data via API calls or web-scraping, loading and operating on data in files, returning data to reactive visualization and dashboards, amongst others. 

In these scenarios you can draw on threading and multi-processing techniques to run tasks concurrently. 

## Synchronous & Asynchronous execution

<div>
    <img src="/assets/images/2022-01-04-multi-tasking/parallel-processing.png" width="90%" height="90%">
</div>

Python code can run in one of two "worlds", synchronous and asynchronous. 

The synchronous world is our typical day to day python. In the synchronous world tasks and jobs are run one after the other on threads. The only option to multi-task here is to use multiple execution threads - covered below in [Other Methods](#other-methods).

The asynchronous world is in a whole different space with different libraries and routines. In asynchronous tasks are run concurrently on a central [event loop](https://docs.python.org/3/library/asyncio-eventloop.html). The event loop is the core of every asyncio application. Event loops run asynchronous tasks and callbacks, perform network IO operations, and run sub-processes.


Given the two worlds there are a few different execution patterns with which we call synchronous and asynchronous functions: 

1. Sync: This is just regular python functions that run sequentially.
2. Async: Use async functions that can be run as concurrent tasks. 
3. Async execution of Sync: We don't want to be limited to just using async specific functions. In some cases it is possible to run sync functions asynchronously.

We don't need to cover Sync here so lets jump straight to point 2 Async.

### Async 

Async functions can be used directly with `await/async` functionality. 

Below is a example async task "coroutine" to sleep. Note that instead of using the standard `time.sleep()` it uses the async compatible `asyncio.sleep()` function.

> **NOTE Jupyter Notebooks**: 
> 
> If you're using IPython or Jupyter Notebooks, you must invoke `async main()` functions with `await main()`. If you're using python shell use `asyncio.run(main())`. 
>   ```
> # example.py
>   import asyncio
>   import time
> 
>   ## multiple task
>   async def say_hello(name):
>       await asyncio.sleep(3)
>       print("Hello-%s" % name)
>
>    async def main():
>        await say_hello("Ben")
>        await say_hello("Jenny")
>
>    start_time = time.time()
>    asyncio.run(main())
>    print(f"--- {time.time() - start_time:.5f} seconds ---\n\r")
>    ```


```python
import asyncio

# single task
async def say_hello(name):
    await asyncio.sleep(3)
    print("Hello-%s" % name)

async def main():
    await say_hello("Ben")

start_time = time.time()
await main()
print(f"--- {time.time() - start_time:.5f} seconds ---\n\r")
```
    Hello-Ben
    --- 3.00425 seconds ---

OK, now let's add another task.


```python
## multiple task
async def say_hello(name):
    await asyncio.sleep(3)
    print("Hello-%s" % name)

async def main():
    await say_hello("Ben")
    await say_hello("Jenny")

start_time = time.time()
await main()
print(f"--- {time.time() - start_time:.5f} seconds ---\n\r")
```
    Hello-Ben
    Hello-Jenny
    --- 6.00553 seconds ---

But hang on, that took 6 seconds. The tasks are still being executed in sync. 

This is expected as, although we are using asynchronous functions, we are still "awaiting" on them to complete sequentially "in sync". 

The [asyncio concurrency documentation](https://docs.python.org/3/library/asyncio-task.html) explains the primary two ways to execute concurrent tasks.

First, we can explicitly build await coroutines.


```python
## multiple task
async def say_hello(name):
    await asyncio.sleep(3)
    print("Hello-%s" % name)

async def main():
    first_task = asyncio.create_task(say_hello("Ben"))
    second_task = asyncio.create_task(say_hello("Jenny"))
    await first_task
    await second_task

start_time = time.time()
await main()
print(f"--- {time.time() - start_time:.5f} seconds ---\n\r")
```
    Hello-Ben
    Hello-Jenny
    --- 3.00403 seconds ---

Neat, now our two 3 second operations are completed in 3 seconds. Not 6.

The second way  to declare async/await coroutines is using [waiting primitives](https://docs.python.org/3/library/asyncio-task.html#waiting-primitives). 

Below is another simple waiting function, but also with an Input/Output pattern and using `asyncio.wait()` to create each task dynamically. 

Note how the total time taken to execute the tasks is equal to the highest time the function waits for. There is also a timeout argument - and this might be handy if making API or web scraping calls for instance. Try changing it to a value less that 9. 


```python
list_items = list()

async def slow_calculation(wait=5):
    global list_items
    print(f'Waiting {wait}....')
    await asyncio.sleep(wait)
    list_items.append(wait)

async def main():
     await asyncio.wait([
                         slow_calculation(i) for i in np.arange(1,10)
                        ],timeout=10)

start_time = time.time()
await main()
print(f"--- {time.time() - start_time:.5f} seconds ---\n\r")

print(f"List items:\t{list_items}")
```

    Waiting 9....
    Waiting 2....
    Waiting 4....
    Waiting 8....
    Waiting 3....
    Waiting 7....
    Waiting 5....
    Waiting 1....
    Waiting 6....
    --- 9.00256 seconds ---
    
    List items:	[1, 2, 3, 4, 5, 6, 7, 8, 9]

### Sync to Async

Here is a simple sync function to sleep - as you can see it takes some time to run each task.


```python
list_items = list()

def slow_calculation(wait=5):
    global list_items
    print(f'Waiting {wait}....')
    time.sleep(wait)
    list_items.append(wait)

def main():
    [slow_calculation(i) for i in np.arange(1,5)]

start_time = time.time()
main()
print(f"--- {time.time() - start_time:.5f} seconds ---\n\r")

print(f"List items:\t{list_items}")
```

    Waiting 1....
    Waiting 2....
    Waiting 3....
    Waiting 4....
    --- 10.00870 seconds ---
    
    List items:	[1, 2, 3, 4]

And here, we use a combination of `ThreadPool` execution and `async` to run each task concurrently using threads. 

The `workers` specifies the number of threads to use - try setting this to 1. 

Alternatively, set the executor argument to `None` in `loop.run_in_executor(None,...` to default to the number of processors on the machine, multiplied by 5 [(documentation)](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor).


```python
from concurrent.futures import ThreadPoolExecutor

workers = 10
list_items = list()

def slow_calculation(wait=5):
    global list_items
    print(f'Waiting {wait}....')
    time.sleep(wait)
    list_items.append(wait)

async def main():
    executor = ThreadPoolExecutor(max_workers=workers)
    loop = asyncio.get_event_loop()
    futures =  [loop.run_in_executor(executor, slow_calculation, i) for i in np.arange(1,5)]
    # futures =  [loop.run_in_executor(None, slow_calculation, i) for i in np.arange(1,5)]
    await asyncio.gather(*futures)

start_time = time.time()
await main()
print(f"--- {time.time() - start_time:.5f} seconds ---\n\r")
print(f"List items:\t{list_items}")
```
    Waiting 1....
    Waiting 2....
    Waiting 3....
    Waiting 4....
    --- 4.00656 seconds ---
    
    List items:	[1, 2, 3, 4]

That concludes this introduction to asynchronous python routines.

## Other methods

Here are some other ways to multi-task in python. 

### threading

The `threading` package is used to enable more explicit control of a thread process. This approach can be useful when waiting on the result of a slow or external process such as an API call or database query. 

A `slow_calculation` function updating a global variable `result` and `thread.join()` is used to ensure that the slow function completes before progressing. 


```python
import threading
result = None

def slow_calculation():
    
    # here goes some long calculation
    rand = np.random.randint(low=2,high=8)
    print(f'Waiting {rand}....')
    time.sleep(rand)
   
    # when the calculation is done, the result is stored in a global variable
    global result
    result = rand**2

def main():
    thread = threading.Thread(target=slow_calculation())
    thread.start()
 
    # dont do this
    # while result is None:
    #     pass
    
    # Do this, wait here for the result to be available before continuing
    thread.join()
   
    print('The result is', result)
    
main()
```
    Waiting 3....
    The result is 9

Below is a similar, but this time instead of waiting for the whole slow function to complete we use an event `result_available = threading.Event()` to trigger the continuation of the thread. 


```python
result = None
result_available = threading.Event()

def slow_calculation():
    
    # here goes some long calculation
    rand = np.random.randint(low=2,high=8)
    print(f'Waiting {rand}....')
    time.sleep(rand)
   
    # when the calculation is done, the result is stored in a global variable
    global result
    result = rand**2
    result_available.set()
    
    # do some more work before exiting the thread
    time.sleep(2)
    print('thread finished')

def main():
    thread = threading.Thread(target=slow_calculation())
    thread.start()
 
    # wait here for the result to be available before continuing
    result_available.wait()
   
    print('The result is', result)
    
main()
```
    Waiting 4....
    thread finished
    The result is 16

Finally, let's look at running multiple threads concurrently now that we understand how threads operate and can be controlled. 

```python
import concurrent.futures
import numpy as np
import time

list_items=list()

def slow_calculation(i):
    # here goes some long calculation
    print(f'Waiting {i}....')
    time.sleep(i)
   
    # when the calculation is done, the result is stored in a global variable
    global list_items
    list_items.append(i**2)

start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(slow_calculation, np.arange(1,5))

print(list_items)
print(f"--- {time.time() - start_time:.5f} seconds ---\n\r")
```
    Waiting 1....
    Waiting 2....
    Waiting 3....
    Waiting 4....
    [1, 4, 9, 16]
    --- 4.00680 seconds --

### multiprocessing

The [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) package supports spawning processes similar to the threading module. Multiprocessing does this by spawning sub-processes that allows users to leverage multiple processors at simultaneously.


```python
from multiprocessing import Pool

list_items=list()

def slow_calculation(wait=5):
    global list_items
    print(f'Waiting {wait}....')
    time.sleep(wait)
    return wait

start_time = time.time()

result_objs = []
with Pool(processes=os.cpu_count() - 1) as pool:
    for i in np.arange(1,5):
        result = pool.apply_async(slow_calculation, (i,))
        result_objs.append(result)
    list_items = [result.get() for result in result_objs]

print(f"--- {time.time() - start_time:.5f} seconds ---\n\r")
print(f"List items:\t{list_items}")
```

    Waiting 1....Waiting 4....
    
    Waiting 3....
    Waiting 2....
    --- 4.16348 seconds ---
    
    List items:	[1, 2, 3, 4]

### pyrallel

[PyRallel](https://pyrallel.readthedocs.io/en/latest/parallel_processor.html) is another ParallelProcessor that uses CPU cores to process compute-intensive tasks. It is similar to multiprocessing.

There is relatively little/no documentation on PyRallel beyond the [official documentation](https://pyrallel.readthedocs.io/en/latest/parallel_processor.html). There is also an unrelated and deprecated predecessor to Dask with the [same name](https://github.com/pydata/pyrallel) - so expect little help. But it also has as a `MapReduce` and `Queue` function that looks interesting and that i'll check out at some point.

 To install:

``` 
# conda 
# from the auto channel https://anaconda.org/auto/pyrallel
conda install -c auto pyrallel

# pip 
# https://pyrallel.readthedocs.io/en/latest/installation.html
pip install pyrallel.lib
```


```python
from pyrallel import ParallelProcessor

list_items=list()

def slow_calculation(wait=5):
    print(f'Waiting {wait}....')
    time.sleep(wait)
    return wait

def collector(data):
    global list_items
    list_items.append(data)

start_time = time.time()

pp = ParallelProcessor(num_of_processor=4,mapper=slow_calculation,collector=collector)
pp.start()
pp.map([i for i in np.arange(1,5)])
pp.task_done()
pp.join()

print(f"--- {time.time() - start_time:.5f} seconds ---\n\r")
print(f"List items:\t{list_items}")
```

    Waiting 1....
    Waiting 4....
    Waiting 2....
    Waiting 3....
    --- 5.20214 seconds ---
    
    List items:	[1, 2, 3, 4]



Thank you for reading

## References

- https://www.aeracode.org/2018/02/19/python-async-simplified/
- https://e2eml.school/multiprocessing.html
- https://pyrallel.readthedocs.io/en/latest/parallel_processor.html
- https://www.aeracode.org/2018/02/19/python-async-simplified/
