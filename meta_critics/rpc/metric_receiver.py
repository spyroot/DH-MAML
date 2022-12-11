import threading
import traceback

from queue import Queue, Full, Empty
from typing import Dict

import torch


class MetricReceiver:

    def __init__(self, num_episodes):
        self.ls_steps_metric = torch.empty(num_episodes)
        self.step = 0

        # buffer called from trainer,   cv used to notify data arrived.
        self.buffer = Queue(maxsize=1)
        self.main_buffer = Queue(maxsize=1)
        self.self_cv = threading.Condition()
        self.self_main_cv = threading.Condition()

        self.shutdown_flag = threading.Event()

        self.__thread1 = threading.Thread(target=self.start_producer,
                                          args=(self.buffer, self.self_cv, self.self_main_cv))
        self.__thread1.daemon = True
        self.__thread1.start()

        self.__thread2 = threading.Thread(target=self.start_consumer, args=(self.buffer, self.self_cv))
        self.__thread2.daemon = True
        self.__thread2.start()

    def shutdown(self):
        """
        :return:
        """
        try:
            self.shutdown_flag.set()
            self.__thread1.daemon = False
            self.__thread2.daemon = False
            self.__thread1.join()
            self.__thread2.join()
        except Exception as err:
            print("Failed to join", err)

    def start_producer(self, buffer, cv, main):
        """
        :return:
        """
        self.buffer = buffer
        self.self_cv = cv
        self.self_main_cv = main
        try:
            self.producer()
        except Exception as err:
            print(err)
            print(traceback.print_exc())

    def start_consumer(self, buffer, cv):
        """
        :return:
        """
        self.buffer = buffer
        self.self_cv = cv
        try:
            self.consumer()
        except Exception as err:
            print(err)
            print(traceback.print_exc())

    def update(self, data: Dict):
        """Receive a data from upstream thread, many threads can push.
        :param data:
        :return:
        """

        if self.main_buffer.full():
            print("Can't keep up. give up")
            # self.self_cv.notify_all()
            with self.self_main_cv:
                self.self_main_cv.notify()
            return

        # self.main_buffer.put_nowait(data)

        with self.self_main_cv:
            try:
                self.main_buffer.put_nowait(data)
                self.self_main_cv.notify()
            except Exception as err:
                print("error")
            finally:
                print("Return")
                self.self_main_cv.notify()

    def producer(self):
        """Receive a data from upstream thread, many threads can push.
        :return:
        """
        while not self.shutdown_flag.is_set():
            # if data empty sleep
            with self.self_main_cv:
                data = None
                try:
                    data = self.main_buffer.get_nowait()
                    if data is None:
                        continue
                    self.main_buffer.task_done()
                    self.self_main_cv.notify()
                except Empty as empty:
                    print("Producer sleeping main queue empty.")
                    self.self_main_cv.notify_all()
                    self.self_main_cv.wait()
                    pass
                finally:
                    self.self_main_cv.notify_all()

            if data is not None:
                with self.self_cv:
                    try:
                        self.buffer.put_nowait(data)
                        self.self_cv.notify()
                    except Full as full:
                        print("Producer sleeping, Consumer queue full")
                        self.self_cv.notify()
                        self.self_cv.wait()
                        pass
                # self.self_cv.notify()

        print("Shutdown event.")

    def consumer(self):
        """
        :return:
        """
        while not self.shutdown_flag.is_set():
            try:
                with self.self_cv:
                    print("Consumer wake up")
                    data = self.buffer.get_nowait()
                    if data is None:
                        with self.self_cv:
                            self.self_cv.notify()
                            self.self_cv.wait()

                self.buffer.task_done()
                print("consumer data", data)

                if 'ls_step' in data:
                    self.ls_steps_metric[self.step] = data['ls_step']

            except Empty as empty:
                with self.self_cv:
                    print("Consumer no data, sleeping")
                    self.self_cv.notify()
                    self.self_cv.wait()

        print("Shutdown event.")


