import time

def timedHDU(func):
    def wrapper(*arg, **kw):
        t1_wall = time.time()
        t1 = time.clock()
        res = func(*arg, **kw)
        t2 = time.clock()
        hdu = res[-1] if type(res) is list else res
        hdu.header['cputime'] = t2-t1
        hdu.header['walltime'] = time.time()-t1_wall
        return res
    return wrapper
