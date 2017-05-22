#coding=utf8
'''
    used to config the logging
'''
import logging

#logging.basicConfig(level=logging.DEBUG)

#logger = logging.getLogger(__name__)
#
#logger.info('start read database')
#
#records = {'john':55, 'tom': 66}
#logger.debug('records: %s', records)
#logger.info('updating records')
#logger.info('finish updating')

import time
import re
import os
import sys
import stat
import logging
import logging.handlers as handlers

class SizedTimedRotatingFileHandler(handlers.TimedRotatingFileHandler):
    """
    Handler for logging to a set of files, which switches from one file
    to the next when the current file reaches a certain size, or at certain
    timed intervals
    """
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None,
                 delay=0, when='h', interval=1, utc=False):
        # If rotation/rollover is wanted, it doesn't make sense to use another
        # mode. If for example 'w' were specified, then if there were multiple
        # runs of the calling application, the logs from previous runs would be
        # lost if the 'w' is respected, because the log file would be truncated
        # on each run.
        if maxBytes > 0:
            mode = 'a'
        handlers.TimedRotatingFileHandler.__init__(
            self, filename, when, interval, backupCount, encoding, delay, utc)
        self.maxBytes = maxBytes

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.

        Basically, see if the supplied record would cause the file to exceed
        the size limit we have.
        """
        if self.stream is None:                 # delay was set...
            self.stream = self._open()
        if self.maxBytes > 0:                   # are we rolling over?
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)  #due to non-posix-compliant Windows feature
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return 1
        t = int(time.time())
        if t >= self.rolloverAt:
            return 1
        return 0

def demo_SizedTimedRotatingFileHandler():
    log_filename='test_log/log_rotate'
    logger=logging.getLogger('MyLogger')
    logger.setLevel(logging.DEBUG)
    handler=SizedTimedRotatingFileHandler(
        log_filename, maxBytes=100, backupCount=5,
        when='s',interval=10,
        # encoding='bz2',  # uncomment for bz2 compression
        )
    logger.addHandler(handler)
    for i in range(10000):
        time.sleep(0.1)
        logger.debug('i=%d' % i)

def init_logger(logger_name='', log_file='', log_level='', print_console=False):

    logger = logging.getLogger()
    logger.setLevel(log_level)

    # create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)


    error_handler = logging.StreamHandler(sys.stdout)
    error_handler.setLevel(logging.ERROR)

    # create a logging format
    formatter = logging.Formatter('%(name)s-logging.%(levelname)s-%(thread)d-%(asctime)s-%(message)s')
    handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    daily_handler=SizedTimedRotatingFileHandler(log_file, when='midnight')
    daily_handler.setLevel(logging.INFO)
    daily_handler.setFormatter(formatter)

    # add the handlers to the logger
    #logger.addHandler(handler)
    logger.addHandler(error_handler)
    logger.addHandler(daily_handler)

    if print_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

if __name__ == '__main__':
    #demo_SizedTimedRotatingFileHandler()
    logger = init_logger('test', 'test.log', logging.INFO, True)
    #logger.info('Hello baby')
    #logger.error('Hello, I am an error!')
    #logger.critical('Hello, I am a critical!')
    logger.info('Hello, I am a warning!')
