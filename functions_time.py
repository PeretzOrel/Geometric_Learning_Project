import timeit

def tic():
    return timeit.default_timer()


def toc(start):
    stop = timeit.default_timer()
    print('Time: ', stop - start)