from contextlib import contextmanager
import io, time

@contextmanager
def log_time():
    start = time.time()
    buff = io.StringIO()
    yield buff
    end = time.time()
    buff.seek(0)
    print("in %.3f sec. "%(end - start) + buff.read())

if __name__ == '__main__':
    with log_time() as log:
        time.sleep(1)
        log.write('hellow')
