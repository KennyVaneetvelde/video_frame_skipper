from collections import namedtuple
import subprocess

VideoMetadata = namedtuple("VideoMetadata", ["frame_count", "fps", "width", "height"])
VideoProcessingState = namedtuple(
    "VideoProcessingState",
    ["total_duration", "skipped_duration", "previous_frame"],
)


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
