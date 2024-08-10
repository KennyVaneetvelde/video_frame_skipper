import cv2
from threading import Thread, Event
from queue import Queue, Empty

class VideoWriter:
    def __init__(self, output_file, fourcc, fps, width, height):
        self.writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        self.stopped = Event()
        self.queue = Queue(maxsize=1024)
        self.thread = Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        while not self.stopped.is_set():
            try:
                frames = self.queue.get(timeout=1)
                if frames is None:
                    break
                for frame in frames:
                    self.writer.write(frame)
            except Empty:
                continue

    def write_batch(self, batch):
        self.queue.put(batch)

    def stop(self):
        self.stopped.set()
        self.queue.put(None)  # Signal end of writing
        self.thread.join()
        self.writer.release()