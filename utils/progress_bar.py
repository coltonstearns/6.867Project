import sys

"""
A basic progress bar implementation, starter code from:
https://stackoverflow.com/questions/6169217/replace-console-output-in-python
"""

class ProgressBar:
    def __init__(self, title, job_length, batch_size):
        self.job_length = job_length
        self.batch_size = batch_size 
        self.progress = 0
        self.title = title
        self.interrupt_flag = False
    
    def make_progress(self, bar_length = 10):
        self.progress += self.batch_size
        percent = float(self.progress) / self.job_length
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\rPercent: [{0}] {1}% {2}/{3}".format(arrow + spaces, int(round(percent * 100)), self.progress, self.job_length))
        sys.stdout.flush()
    
