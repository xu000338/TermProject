"""
logger.py
CST 2213
Project
David Xu (041173885)    
Description:
Contains code for exception handling and logging methods
"""
  
#import logging module
import logging

import functools
from functools import reduce

def new_logger():
    logger=logging.getLogger("Program started! \n Logger activated!")
    logger.setLevel(logging.INFO)
    #create the logfile
    logfile = logging.FileHandler("logger.log")
    fmt = '%(asctime)s - %(name)s - %(levelname)s 0 %(message)s'
    formatter = logging.Formatter(fmt)
    logfile.setFormatter(formatter)
    #add logfile to handler
    logger.addHandler(logfile)
    return logger
    
def exception_decorator(func):

    @functools.wraps(func)
    def log_wrapper(*args, **kwargs):
        
        #create the logger
        logger = new_logger()
        try:
            return func(*args, **kwargs)
        except ValueError:
            #log the exception
            error=f"ValueError exception occurred in: '{func.__name__}' "
            print(error)
            logger.exception(error)
        except TypeError:
           #log the exception
            error=f"TypeError exception occurred in: '{func.__name__}' "
            print(error)
            logger.exception(error)
        #for all other exceptions
        except:        
            #log the exception
            error=f"Exception occurred in: '{func.__name__}' - See .log file for details "
            print(error)
            logger.exception(error)
    return log_wrapper
