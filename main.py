import argparse
from video_processor import VideoProcessor
from frame_processor import FrameProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove static frames from a video.")
    parser.add_argument(
        "--input_file",
        default="input_short.mp4",
        help="Path to the input video file (default: input.mp4)",
    )
    parser.add_argument(
        "--output_file",
        default="output2.mp4",
        help="Path for the output video file (default: output.mp4)",
    )

    args = parser.parse_args()

    processor = VideoProcessor(
        args.input_file,
        args.output_file,
    )
    frame_processor = FrameProcessor()
    total_duration, skipped_duration, processed_duration = (
        processor.remove_static_frames(frame_processor)
    )

    print(f"Original duration: {total_duration:.2f} seconds")
    print(f"Skipped duration: {skipped_duration:.2f} seconds")
    print(f"Processed duration: {processed_duration:.2f} seconds")