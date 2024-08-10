import cupy as cp
from utils import VideoProcessingState

class FrameProcessor:
    def __init__(self, kernel_size=7, sigma=3.0, block_size=16, hi_threshold=400, lo_threshold=100, frac_threshold=0.1):
        self.gaussian_kernel = self._gaussian_kernel(kernel_size, sigma)
        self.block_size = block_size
        self.hi_threshold = hi_threshold
        self.lo_threshold = lo_threshold
        self.frac_threshold = frac_threshold

    @staticmethod
    def _gaussian_kernel(size, sigma):
        """Create a 2D Gaussian kernel."""
        x, y = cp.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
        g = cp.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
        return g / g.sum()

    @staticmethod
    def _apply_gaussian_blur(image, kernel):
        """Apply Gaussian blur using CUDA."""
        return cp.real(
            cp.fft.ifft2(cp.fft.fft2(image) * cp.fft.fft2(kernel, s=image.shape))
        )

    def process_batch(self, batch, state, fps):
        # Convert the batch to a CuPy array and move it to GPU
        batch_gpu = cp.asarray(batch)

        # Perform RGB to grayscale conversion on GPU
        gray_frames_gpu = cp.dot(batch_gpu[..., :3], cp.array([0.299, 0.587, 0.114]))

        # Apply Gaussian blur to each frame
        blurred_frames_gpu = cp.array(
            [
                self._apply_gaussian_blur(frame, self.gaussian_kernel)
                for frame in gray_frames_gpu
            ]
        )

        if len(batch) == 1:
            # Handle the case when batch size is 1
            include_frames = cp.array([True])
        else:
            # Compute differences for the frames
            include_frames = self._compute_block_difference(blurred_frames_gpu)

        total_duration = state.total_duration + len(batch) / fps
        skipped_duration = state.skipped_duration + cp.sum(~include_frames) / fps

        return (
            VideoProcessingState(
                total_duration=total_duration,
                skipped_duration=skipped_duration,
                previous_frame=blurred_frames_gpu[-1],
            ),
            include_frames,
            batch_gpu,
        )

    def _compute_block_difference(self, frames):
        height, width = frames.shape[1:]
        blocks_v = height // self.block_size
        blocks_h = width // self.block_size

        # Trim frames to fit block size
        trimmed_height = blocks_v * self.block_size
        trimmed_width = blocks_h * self.block_size
        frames = frames[:, :trimmed_height, :trimmed_width]

        prev_frames = frames[:-1]
        curr_frames = frames[1:]

        # Reshape frames into blocks
        prev_blocks = prev_frames.reshape(prev_frames.shape[0], blocks_v, self.block_size, blocks_h, self.block_size)
        curr_blocks = curr_frames.reshape(curr_frames.shape[0], blocks_v, self.block_size, blocks_h, self.block_size)

        # Calculate Sum of Absolute Differences (SAD) for each block
        sad = cp.abs(curr_blocks - prev_blocks).sum(axis=(2, 4))

        # Count blocks exceeding thresholds
        hi_count = (sad > self.hi_threshold).sum(axis=(1, 2))
        lo_count = (sad > self.lo_threshold).sum(axis=(1, 2))

        # Determine which frames to keep
        total_blocks = blocks_v * blocks_h
        include_frames = (hi_count > 0) | (lo_count > self.frac_threshold * total_blocks)

        # Include the first frame
        return cp.concatenate((cp.array([include_frames[0]]), include_frames))

