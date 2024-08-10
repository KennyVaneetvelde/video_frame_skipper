import cv2
import cupy as cp
from tqdm import tqdm
from video_stream import VideoStream
from video_writer import VideoWriter
from utils import VideoMetadata, VideoProcessingState, extract_video_metadata


class VideoProcessor:
    def __init__(
        self,
        input_file,
        output_file,
        batch_size=32,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size
        self.metadata = extract_video_metadata(input_file)
        self.video_stream = VideoStream(input_file, batch_size).start()
        self.video_writer = VideoWriter(
            output_file,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.metadata.fps,
            self.metadata.width,
            self.metadata.height,
        )

    def remove_static_frames(self, frame_processor):
        state = VideoProcessingState(
            total_duration=0,
            skipped_duration=0,
            previous_frame=None,
        )

        with tqdm(total=self.metadata.frame_count) as pbar:
            while True:
                batch = self.video_stream.read()
                if batch is None:
                    break

                if not batch:  # Skip empty batches
                    continue

                state, include_frames, processed_batch_gpu = (
                    frame_processor.process_batch(batch, state, self.metadata.fps)
                )

                # Filter the batch on GPU
                filtered_batch_gpu = processed_batch_gpu[include_frames]

                # Convert filtered batch back to CPU only when writing
                filtered_batch_cpu = cp.asnumpy(filtered_batch_gpu)
                self.video_writer.write_batch(filtered_batch_cpu)

                pbar.update(len(batch))

        self.video_stream.stop()
        self.video_writer.stop()

        return (
            state.total_duration,
            state.skipped_duration,
            state.total_duration - state.skipped_duration,
        )
