import time

class Timer(object):
    """Basic timer class"""
    def __init__(self, resolution='s'):
        self._start = 0. 
        self._stop = 0.
        if resolution=='s':
            self.order = 1 
        elif resolution=='ms':
            self.order = 1e3 
        elif resolution=='us':
            self.order = 1e6
        elif resolution=='min':
            self.order = 1/60 
        else: raise ValueError(
                'Choose between: '
                '"s" for seconds, ' 
                '"ms" for milliseconds, ' 
                '"us" for microseconds and '
                '"min" for minutes')
        self.resolution = resolution

    def start(self):
        self._start = time.time() 
    
    def stop(self):
        self._stop = time.time()
        elapsed = round((self._stop - self._start) * self.order, 4)
        print(f'\nTraining time: {elapsed} {self.resolution}\n')
        return elapsed