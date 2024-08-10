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
        default="output.mp4",
        help="Path for the output video file (default: output.mp4)",
    )
    parser.add_argument(
        "--difference_threshold",
        type=float,
        default=0.5,
        help="Threshold for frame difference (default: 0.001)",
    )

    args = parser.parse_args()

    processor = VideoProcessor(
        args.input_file,
        args.output_file,
    )
    frame_processor = FrameProcessor(kernel_size=15, sigma=7.0)
    total_duration, skipped_duration, processed_duration = (
        processor.remove_static_frames(frame_processor)
    )

    print(f"Original duration: {total_duration:.2f} seconds")
    print(f"Skipped duration: {skipped_duration:.2f} seconds")
    print(f"Processed duration: {processed_duration:.2f} seconds")