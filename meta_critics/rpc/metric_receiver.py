import threading
import traceback
from queue import Queue, Full, Empty
from typing import Dict
import torch
import wandb

wandb.init(project="dh-maml", entity="spyroot")


class MetricReceiver:

    def __init__(self, num_episodes):
        """

        :param num_episodes:
        """
        self.ls_steps_metric = torch.empty(num_episodes)
        self.step = 0

        # buffer called from trainer,   cv used to notify data arrived.
        self.buffer = Queue(maxsize=100)
        self.main_buffer = Queue(maxsize=100)
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
            print("Failed to join: error", err)

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
            print("Failed to start thread error:", err)
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
            print("Failed to start thread error:", err)
            print(traceback.print_exc())

    def update(self, data: Dict):
        """Receive a data from upstream thread, many threads can push up to main_buffer size.
        If you need absorb more data increase queue size.  Each time data arrived we notify producer
        thread via cv. Produce wake up if there is data it pushes if consumer buffer full it will
        sleep.  While producer sleeping main thread can push up to main queue size and absorb a data.
        :param data:
        :return:
        """

        if self.main_buffer.full():
            print("Can't keep up. give up")
            with self.self_main_cv:
                self.self_main_cv.notify()
            return

        with self.self_main_cv:
            try:
                self.main_buffer.put_nowait(data)
                self.self_main_cv.notify()
            except Exception as err:
                print("Failed take a look", err)
            finally:
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
                except Empty as _:
                    self.self_main_cv.notify_all()
                    self.self_main_cv.wait()
                    pass

            if data is not None:
                with self.self_cv:
                    try:
                        self.buffer.put_nowait(data)
                        self.self_cv.notify()
                    except Full as _:
                        self.self_cv.notify()
                        self.self_cv.wait()
                        pass

        print("Shutdown event.")

    def consumer(self) -> None:
        """ Consumer thread computes all metrics.

        # Training in progress, dev: cpu,:   2%| | 8/499 [06:40<7:40:22, 56.26s/it, inner_pre=-.00939, old_loss=-5.88e-9, old_kl=0, ls_step=15, improved=-2.09e-5, ls_counter=39, inner_postconsumer data {'inner_pre': -0.009393773972988129, 'old_loss': -5.882904385856591e-09, 'old_kl': 0.0, 'ls_step': 15, 'improved': -2.08782876143232e-05, 'ls_counter': 39, 'inner_post': 7.863156497478485e-05, 'loss_post': -2.08782876143232e-05, 'kl_post': 1.6577161659370176e-05, 'train_rewards/meta_train': tensor(-356.1232), 'train_rewards/meta_validation': tensor(-307.9262)}
        :return:
        """
        while not self.shutdown_flag.is_set():
            try:
                with self.self_cv:
                    data = self.buffer.get_nowait()
                    if data is None:
                        with self.self_cv:
                            self.self_cv.notify()
                            self.self_cv.wait()

                self.buffer.task_done()

                # average number of line search steps
                if 'ls_step' in data:
                    self.ls_steps_metric[self.step] = data['ls_step']
                    data['ls_step'] = self.ls_steps_metric.mean()

                wandb.log(data)

            except Empty as _:
                with self.self_cv:
                    self.self_cv.notify()
                    self.self_cv.wait()

        print("Shutdown event.")
