import multiprocessing
import time

def slow_function(n):
    s = 0
    for i in range(n):
        s += n

if __name__ == "__main__":
    n = 50000000

    t0 = time.time()
    p = []
    n_threads = 100
    for i in range(10):
        p.append(multiprocessing.Process(target=slow_function, args=(n,)))
        p[i].start()
    
    for i in range(10):
        p[i].join()
    t1 = time.time()
    
    print(f"parallel time: {t1-t0}")

    t0 = time.time()
    slow_function(n)
    t1 = time.time()
    print(f"single thread time: {t1-t0}")
