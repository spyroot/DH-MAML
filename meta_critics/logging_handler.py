import logging
import multiprocessing
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler


class MultiProcessingLog(logging.Handler):
    def __init__(self, name, mode, maxsize, rotate):
        """

        :param name:
        :param mode:
        :param maxsize:
        :param rotate:
        """
        logging.Handler.__init__(self)
        self._handler = RotatingFileHandler(name, mode, maxsize, rotate)
        self.queue = multiprocessing.Queue(-1)

        t = threading.Thread(target=self.receive)
        t.daemon = True
        t.start()

    def setFormatter(self, fmt):
        """
        :param fmt:
        :return:
        """
        logging.Handler.setFormatter(self, fmt)
        self._handler.setFormatter(fmt)

    def receive(self):
        """
        :return:
        """
        while True:
            try:
                record = self.queue.get()
                self._handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)

    def send(self, s):
        """
        Push to a queue.
        :param s:
        :return:
        """
        self.queue.put_nowait(s)

    def _format_record(self, record):
        """

        :param record:
        :return:
        """
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            dummy = self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        """
        :param record:
        :return:
        """
        try:
            s = self._format_record(record)
            self.send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def close(self):
        """
        :return:
        """
        try:
            self._handler.close()
            logging.Handler.close(self)
        except Exception as err:
            print("Failed to close error: ", err)
