from cmath import log
from logging.config import fileConfig
import logging
import os

class Logger(object):
    instance = None

    def __new__(cls, log_file: str = "logs.log"):
        if not Logger.instance:
            file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), log_file)
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            logging.basicConfig()
            logger = logging.getLogger("project_logs")
            logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(file_path)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            Logger.instance = logger
        return Logger.instance

    def __getattr__(self, name):
        return getattr(Logger.instance, name)

    def __setattr__(self, name):
        return setattr(Logger.instance, name)