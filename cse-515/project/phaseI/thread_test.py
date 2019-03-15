# Thread Testing
import threading


def foo(n):
    num = 0
    for i in range(0,n):
        num += i
    print(str(num))

def spawn_threads(num_threads, n):

    threads = []
    for i in range(0, num_threads):
        this_thread = threading.Thread(target=foo, args=(n,))
        this_thread.start()
        threads.append(this_thread)
    return threads

def thread_state(threads):

    cont = True
    counter = 10
    while cont:
        for i in threads:
            if not i.isAlive():
                if counter <= 0:
                    cont = False
                else:
                    counter -= 1
            
            print(str(i) + " : " + str(i.isAlive()))

thread_state(spawn_threads(2,200))