import cv2
from threading import Thread, Event
from queue import Queue, Empty

class VideoStream:
    def __init__(self, src, batch_size=512):
        self.stream = cv2.VideoCapture(src)
        self.stopped = Event()
        self.queue = Queue(maxsize=1024)
        self.batch_size = batch_size
        self.thread = None

    def start(self):
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        batch = []
        while not self.stopped.is_set():
            if not self.queue.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    if batch:
                        self.queue.put(batch)
                    self.queue.put(None)  # Signal end of stream
                    break
                batch.append(frame)
                if len(batch) == self.batch_size:
                    self.queue.put(batch)
                    batch = []

    def read(self):
        try:
            return self.queue.get(timeout=1)
        except Empty:
            return []

    def stop(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
        self.stream.release()