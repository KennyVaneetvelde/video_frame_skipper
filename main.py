import cv2
import numpy as np
import cupy as cp
import subprocess
import os
from tqdm import tqdm
from collections import namedtuple
from threading import Thread
from queue import Queue

# Set up CUDA device
cp.cuda.Device(0).use()

VideoMetadata = namedtuple("VideoMetadata", ["frame_count", "fps", "width", "height"])
FrameAnalysisResult = namedtuple(
    "FrameAnalysisResult", ["is_significant", "static_duration"]
)
VideoProcessingState = namedtuple(
    "VideoProcessingState",
    ["total_duration", "skipped_duration", "static_duration", "previous_frame"],
)


class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.queue = Queue(maxsize=2048)

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.queue.full():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stop()
                    return
                self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True


class VideoWriter:
    def __init__(self, output_file, codec, fps, width, height):
        self.output_file = output_file
        self.codec = codec
        self.fps = fps
        self.width = width
        self.height = height
        self.writer = None
        self.queue = Queue(maxsize=2048)
        self.stopped = False
        self.thread = Thread(target=self.write_frames)
        self.thread.start()

    def write_frames(self):
        self.writer = cv2.VideoWriter(
            self.output_file, self.codec, self.fps, (self.width, self.height)
        )
        while True:
            if self.stopped and self.queue.empty():
                return
            if not self.queue.empty():
                frame = self.queue.get()
                self.writer.write(frame)

    def write(self, frame):
        self.queue.put(frame)

    def stop(self):
        self.stopped = True
        self.thread.join()
        if self.writer:
            self.writer.release()


def extract_video_metadata(input_file):
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets,r_frame_rate,width,height",
        "-of",
        "csv=p=0",
        input_file,
    ]
    output = subprocess.check_output(ffprobe_cmd).decode("utf-8").strip().split(",")
    return VideoMetadata(
        frame_count=int(output[3]),
        fps=eval(output[2]),
        width=int(output[0]),
        height=int(output[1]),
    )


def compute_frame_difference(frame1, frame2):
    diff = cp.abs(frame1.astype(cp.float32) - frame2.astype(cp.float32))
    return cp.mean(diff)


def analyze_frame(
    current_frame,
    previous_frame,
    difference_threshold,
    static_duration,
    max_static_duration,
    fps,
):
    current_frame_gpu = cp.asarray(current_frame)
    diff = compute_frame_difference(previous_frame, current_frame_gpu)

    if diff < difference_threshold:
        static_duration += 1 / fps
        is_significant = static_duration <= max_static_duration
    else:
        static_duration = 0
        is_significant = True

    return current_frame_gpu, FrameAnalysisResult(is_significant, static_duration)


def capture_frame(video_capture):
    success, frame = video_capture.read()
    return frame if success else None


def update_video_durations(state, fps, is_significant):
    total_duration = state.total_duration + 1 / fps
    skipped_duration = state.skipped_duration + (0 if is_significant else 1 / fps)
    return total_duration, skipped_duration


def initialize_video_processing_state():
    return VideoProcessingState(
        total_duration=0, skipped_duration=0, static_duration=0, previous_frame=None
    )


def process_initial_frame(frame, fps):
    return VideoProcessingState(
        total_duration=1 / fps,
        skipped_duration=0,
        static_duration=0,
        previous_frame=cp.asarray(frame),
    )


def process_video_frame(frame, state, difference_threshold, max_static_duration, fps):
    previous_frame, result = analyze_frame(
        frame,
        state.previous_frame,
        difference_threshold,
        state.static_duration,
        max_static_duration,
        fps,
    )
    total_duration, skipped_duration = update_video_durations(
        state, fps, result.is_significant
    )
    return (
        VideoProcessingState(
            total_duration=total_duration,
            skipped_duration=skipped_duration,
            static_duration=result.static_duration,
            previous_frame=previous_frame,
        ),
        result.is_significant,
    )


def process_video(
    video_stream,
    video_metadata,
    difference_threshold,
    max_static_duration,
    video_writer,
):
    state = initialize_video_processing_state()

    for _ in tqdm(range(video_metadata.frame_count)):
        frame = video_stream.read()
        if frame is None:
            break

        if state.previous_frame is None:
            state = process_initial_frame(frame, video_metadata.fps)
            video_writer.write(frame)
            continue

        state, is_significant = process_video_frame(
            frame, state, difference_threshold, max_static_duration, video_metadata.fps
        )

        if is_significant:
            video_writer.write(frame)

    return state.total_duration, state.skipped_duration


def remove_static_frames(
    input_file, output_file, difference_threshold, max_static_duration
):
    video_metadata = extract_video_metadata(input_file)
    video_stream = VideoStream(input_file).start()
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = VideoWriter(
        output_file,
        codec,
        video_metadata.fps,
        video_metadata.width,
        video_metadata.height,
    )

    total_duration, skipped_duration = process_video(
        video_stream,
        video_metadata,
        difference_threshold,
        max_static_duration,
        video_writer,
    )

    video_stream.stop()
    video_writer.stop()

    return total_duration, skipped_duration, total_duration - skipped_duration


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove static frames from a video.")
    parser.add_argument("input_file", help="Path to the input video file")
    parser.add_argument("output_file", help="Path for the output video file")
    parser.add_argument(
        "--difference_threshold",
        type=float,
        default=0.005,
        help="Threshold for frame difference (default: 0.005)",
    )
    parser.add_argument(
        "--max_static_duration",
        type=float,
        default=0.01,
        help="Maximum duration of static content to keep (default: 0.01)",
    )

    args = parser.parse_args()

    difference_threshold = args.difference_threshold
    max_static_duration = args.max_static_duration
    buffer_size = 2048  # Number of frames to buffer before writing

    total_duration, skipped_duration, processed_duration = remove_static_frames(
        args.input_file, args.output_file, difference_threshold, max_static_duration
    )

    print(f"Original duration: {total_duration:.2f} seconds")
    print(f"Skipped duration: {skipped_duration:.2f} seconds")
    print(f"Processed duration: {processed_duration:.2f} seconds")
