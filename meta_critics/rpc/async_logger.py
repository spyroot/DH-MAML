import asyncio
import logging
import threading
import traceback

from aiologger import Logger


class AsyncLogger:

    self_queue = asyncio.Queue()

    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        super(AsyncLogger, self).__init__(*args, **kwargs)

        self.loop = None
        self.logger = None
        self.__thread = threading.Thread(target=self.start_loop)
        self.__thread.daemon = True
        # self.__thread.start()

    async def emit_async(self, record):
        await AsyncLogger.self_queue.put(record)

    def emit(self, record: str):
        AsyncLogger.self_queue.put_nowait(record)
        if self.loop is not None:
            self.loop._write_to_self()

    def start_loop(self):
        try:
            self.loop = asyncio.new_event_loop()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.logger = Logger.with_default_handlers(name='meta_critic', level=logging.INFO)
            loop.run_until_complete(self.consumer())
            loop.close()
        except Exception as thread_exception:
            print("Thread exception", thread_exception)
            print(traceback.print_exc())

    async def consumer(self):
        """
        :return:
        """
        try:
            print("Consumer started.")
            await self.sink()
            print("Consumer shutting down logger.")
            await self.logger.shutdown()
            print("Consumer stopped.")
        except Exception as consumer_exception:
            print("Thread exception", consumer_exception)
            print(traceback.print_exc())

    async def sink(self):
        """
        :return:
        """
        while True:
            try:
                # wait for an item from the agent
                msg = await AsyncLogger.self_queue.get()
                if msg is None:
                    break

                await self.logger.info(f"{threading.get_native_id()}: {msg}")
                AsyncLogger.self_queue.task_done()

            except Exception as task_exp:
                print(f"{threading.get_native_id()}{task_exp}")
                print(traceback.print_exc())

