import multiprocessing
import time

def slow_function(n, q):
    s = 0
    for i in range(n):
        s += n
    q.put(s)  # send result back

if __name__ == "__main__":
    n = 50000000
    q = multiprocessing.Queue()

    t0 = time.time()
    p = []
    for i in range(10):
        proc = multiprocessing.Process(target=slow_function, args=(n, q))
        p.append(proc)
        proc.start()
    
    for proc in p:
        proc.join()

    results = [q.get() for _ in range(10)]
    t1 = time.time()

    print(f"parallel time: {t1-t0}")
    print(f"results: {results}")

    t0 = time.time()
    single = slow_function(n, multiprocessing.Queue())  # won't return, just example
    t1 = time.time()
    print(f"single thread time: {t1-t0}")
