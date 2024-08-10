# Video Frame Skipper

## Overview

Video Frame Skipper is a Python-based tool that removes static frames from videos, effectively reducing the duration of videos with minimal content changes. It uses GPU acceleration via CUDA to process videos efficiently.

## How It Works

1. **Video Input**: The tool reads the input video file using OpenCV.
2. **Frame Processing**: 
   - Frames are processed in batches for efficiency.
   - Each frame is converted to grayscale and blurred using a Gaussian filter.
   - The tool compares consecutive frames to detect changes.
3. **Change Detection**:
   - The image is divided into blocks.
   - Sum of Absolute Differences (SAD) is calculated for each block between consecutive frames.
   - Blocks are classified as changed based on high and low thresholds.
4. **Frame Selection**: 
   - Frames with significant changes are kept.
   - Static frames are skipped.
5. **Video Output**: The selected frames are written to a new video file.

## Noise Handling

The tool employs several techniques to handle noise and avoid false positives:

1. **Gaussian Blur**: Applied to each frame to reduce small-scale noise.
2. **Block-based Analysis**: The frame is divided into blocks, making the algorithm less sensitive to small, localized changes.
3. **Dual Thresholds**: Uses both high and low thresholds to detect changes, allowing for detection of both large and subtle changes.
4. **Fractional Threshold**: A minimum fraction of changed blocks is required to consider a frame as changed, reducing sensitivity to minor noise.

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU
- CUDA Toolkit (version 12.x recommended)

### Steps

1. Clone the repository:
   ```
   git clone https://github.com/KennyVaneetvelde/video_frame_skipper.git
   cd video-frame-skipper
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Install CUDA:
   - Download and install the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
   - Make sure to install a version compatible with the cupy-cuda12x package in the requirements.txt file.

## Usage

### Command Line

Run the script from the command line with the following arguments:

```
python main.py --input_file <path_to_input_video> --output_file <path_to_output_video>
```

Example:
```
python main.py --input_file input.mp4 --output_file output.mp4
```

### Batch File (Windows)

For Windows users, a batch file `processvideo.bat` is provided for ease of use:

1. Double-click on `processvideo.bat`.
2. When prompted, enter the full path to your input video file.
3. When prompted, enter the full path for the desired output video file.
4. The script will run and process the video.

## Configuration

You can adjust the following parameters in `frame_processor.py` to fine-tune the frame skipping behavior:

- `kernel_size`: Size of the Gaussian blur kernel (default: 7)
- `sigma`: Standard deviation for Gaussian blur (default: 3.0)
- `block_size`: Size of blocks for change detection (default: 16)
- `hi_threshold`: High threshold for block change detection (default: 400)
- `lo_threshold`: Low threshold for block change detection (default: 100)
- `frac_threshold`: Fraction of changed blocks required (default: 0.01)

## Output

The script will display the following information after processing:

- Original video duration
- Skipped duration (time removed)
- Processed duration (final video length)

The processed video will be saved to the specified output file path.

## Limitations

- The tool requires a CUDA-capable GPU for operation.
- Very large videos may require significant processing time and GPU memory.
- The output video uses the MP4V codec, which may not be compatible with all video players.

## Contributing

Contributions to improve the Video Frame Skipper are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

Copyright 2024 Kenny Vaneetvelde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.