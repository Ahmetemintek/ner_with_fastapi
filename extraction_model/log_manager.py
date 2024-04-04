import logging
import os
import sys

logger= None

def init_log(name, log_level=logging.INFO):
    """
    Initialize the logger
    """

    global logger
    if logger is None:
        logger= logging.getLogger(name)
        logger.setLevel(log_level)

        #create a console handler and set the level
        ch= logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)

        #create a formatter
        formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        #add formatter to ch
        ch.setFormatter(formatter)

        #add ch to logger
        logger.addHandler(ch)

    return logger
