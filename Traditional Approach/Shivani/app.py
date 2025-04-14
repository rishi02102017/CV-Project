# -*- coding: utf-8 -*-
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import tempfile
import os

class LaneDetector:
    def __init__(self):
        # Parameters for lane detection
        self.height = None
        self.width = None
        self.roi_vertices = None

    def preprocess_image(self, image):
        """Preprocess the image: resize, convert to appropriate colorspace"""
        # Store image dimensions
        self.height, self.width = image.shape[:2]

        # Improved ROI - wider at top to capture more of the lane in the distance
        self.roi_vertices = np.array([
           [(self.width * 0.1, self.height),             # Bottom left
            (self.width * 0.45, self.height * 0.6),      # Top left
            (self.width * 0.55, self.height * 0.6),      # Top right
            (self.width * 0.9, self.height)]             # Bottom right
        ], dtype=np.int32)


        # Convert to different colorspaces for better lane detection
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        return image, hls

    def color_threshold(self, hls_image):
        """Apply color thresholding to isolate white and yellow lane markings"""
        # White color mask - more permissive for white detection
        lower_white = np.array([0, 190, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hls_image, lower_white, upper_white)

        # Yellow color mask - slightly adjusted for better detection
        lower_yellow = np.array([15, 38, 115], dtype=np.uint8)
        upper_yellow = np.array([35, 204, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hls_image, lower_yellow, upper_yellow)

        # Create additional mask for white in RGB space for better detection of white dashed lines
        rgb_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        _, white_binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

        # Combine all masks
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(white_mask, yellow_mask), white_binary)
        return combined_mask

    def sobel_edge_detection(self, image):
        """Apply Sobel operator for edge detection (alternative to Canny)"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Sobel operator in x direction (to detect vertical edges)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Apply Sobel operator in y direction (to detect horizontal edges)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobely = np.absolute(sobely)
        scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))

        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)

        # Apply threshold
        threshold_min = 30
        threshold_max = 100
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= threshold_min) & (gradmag <= threshold_max)] = 1

        return binary_output

    def apply_roi_mask(self, binary_image):
        """Apply region of interest mask"""
        mask = np.zeros_like(binary_image)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        masked_image = cv2.bitwise_and(binary_image, mask)
        return masked_image

    def perspective_transform(self, image):
        """Apply perspective transform to get bird's eye view"""
        # Define source points - adjusted to better capture lane lines
        # Use a narrower trapezoid at the bottom, wider at the top
        bottom_width = self.width * 0.7  # smaller trapezoid base
        top_width = self.width * 0.2     # narrower top
        height_pct = 0.62                # horizon lower

        src = np.float32([
            [(self.width - bottom_width) / 2, self.height - 10],
            [(self.width - top_width) / 2, self.height * height_pct],
            [(self.width + top_width) / 2, self.height * height_pct],
            [(self.width + bottom_width) / 2, self.height - 10]
        ])

        dst = np.float32([
            [self.width * 0.25, self.height],
            [self.width * 0.25, 0],
            [self.width * 0.75, 0],
            [self.width * 0.75, self.height]
        ])


        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        # Apply perspective transform
        warped = cv2.warpPerspective(image, M, (self.width, self.height), flags=cv2.INTER_LINEAR)

        return warped, M, Minv

    def sliding_window_lane_detection(self, binary_warped):
        """Detect lane lines using sliding window approach"""
        # Take a histogram of the bottom quarter of the image - focus on the closest part of the road
        bottom_quarter = binary_warped.shape[0] * 3 // 4
        histogram = np.sum(binary_warped[bottom_quarter:, :], axis=0)

        # Apply smoothing to the histogram to reduce noise
        histogram = np.convolve(histogram, np.ones(5)/5, mode='same')

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Find the peak of the left and right halves of the histogram
        midpoint = np.int32(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Number of sliding windows - increased for finer resolution
        nwindows = 12
        # Height of windows
        window_height = np.int32(binary_warped.shape[0] / nwindows)

        # Identify all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Width of the windows +/- margin - increased to capture more pixels
        margin = 120
        # Minimum number of pixels found to recenter window - reduced threshold
        minpix = 30

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # If no window centers found, just use bottom ones
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else []
        lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else []
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else []
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else []

        # Fit a second order polynomial to each
        left_fit = None
        right_fit = None
        if len(leftx) > 0 and len(lefty) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 0 and len(righty) > 0:
            right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit, out_img, leftx, lefty, rightx, righty

    def generate_lane_pixels(self, binary_warped, left_fit, right_fit):
        """Generate x and y values for plotting"""
        # Generate y values for plotting regardless of lane detection success
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

        # If either lane fit is None, return None for both fitx values but still return ploty
        if left_fit is None or right_fit is None:
            return None, None, ploty

        # Calculate lane line points if we have both fits
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        return left_fitx, right_fitx, ploty

    def draw_lane_on_original(self, original_image, binary_warped, left_fitx, right_fitx, ploty, Minv):
        """Draw the lane onto the original image"""
        if left_fitx is None or right_fitx is None:
            return original_image

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(original_image, 1, newwarp, 0.2, 0)

        return result

    def process_image(self, image):
        """Process image through the lane detection pipeline"""
        # Preprocess image
        original, hls = self.preprocess_image(image)

        # Apply color thresholding
        color_binary = self.color_threshold(hls)

        # Apply Sobel edge detection
        edge_binary = self.sobel_edge_detection(image)

        # Combine color and edge binary images
        combined_binary = np.zeros_like(color_binary)
        combined_binary[(color_binary > 0) | (edge_binary > 0)] = 1

        # Apply ROI mask
        masked_binary = self.apply_roi_mask(combined_binary)

        # Apply perspective transform
        warped_binary, M, Minv = self.perspective_transform(masked_binary)

        # Detect lane lines using sliding window
        left_fit, right_fit, out_img, leftx, lefty, rightx, righty = self.sliding_window_lane_detection(warped_binary)

        # Generate lane pixels
        left_fitx, right_fitx, ploty = self.generate_lane_pixels(warped_binary, left_fit, right_fit)

        # Draw lane on original image
        result = self.draw_lane_on_original(original, warped_binary, left_fitx, right_fitx, ploty, Minv)

        return result, {
            'original': original,
            'color_binary': color_binary,
            'edge_binary': edge_binary * 255,  # Scale for visualization
            'combined_binary': combined_binary * 255,
            'masked_binary': masked_binary * 255,
            'warped_binary': warped_binary * 255,
            'sliding_window': out_img,
            'result': result
        }


def process_video(video_path, output_path):
    """
    Process a video file for lane detection.
    
    Args:
        video_path: Path to the input video file
        output_path: Path where the processed video will be saved
    """
    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {video_path}")
        return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info(f"Processing video with {frame_count} frames at {fps} fps")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize lane detector
    lane_detector = LaneDetector()

    # Add progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process frames
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Process frame
            result, _ = lane_detector.process_image(frame)

            # Write processed frame
            out.write(result)
        except Exception as e:
            st.warning(f"Error processing frame {frame_idx}: {e}")
            # Use original frame if processing fails
            out.write(frame)

        # Update progress
        frame_idx += 1
        progress = frame_idx / frame_count
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_idx}/{frame_count} ({progress*100:.1f}%)")

    # Release resources
    cap.release()
    out.release()

    return output_path


def display_pipeline_steps(pipeline_images):
    """Display each step of the lane detection pipeline in Streamlit"""
    st.subheader("Pipeline Steps")
    
    # Original image
    st.image(cv2.cvtColor(pipeline_images['original'], cv2.COLOR_BGR2RGB), 
             caption='Original Image', use_column_width=True)
    
    # Create columns for the pipeline steps
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(pipeline_images['color_binary'], 
                 caption='Color Threshold', use_column_width=True)
        st.image(pipeline_images['combined_binary'], 
                 caption='Combined Binary', use_column_width=True)
    
    with col2:
        st.image(pipeline_images['edge_binary'], 
                 caption='Sobel Edge Detection', use_column_width=True)
        st.image(pipeline_images['masked_binary'], 
                 caption='ROI Masked', use_column_width=True)
    
    with col3:
        st.image(pipeline_images['warped_binary'], 
                 caption='Birds Eye View', use_column_width=True)
        st.image(pipeline_images['sliding_window'], 
                 caption='Sliding Window Detection', use_column_width=True)
    
    # Final result
    st.image(cv2.cvtColor(pipeline_images['result'], cv2.COLOR_BGR2RGB), 
             caption='Final Result', use_column_width=True)


def main():
    st.title("Advanced Lane Detection System")
    st.write("Upload an image or video to detect lane markings")

    # Initialize lane detector
    lane_detector = LaneDetector()

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Image Processing", "Video Processing"])

    with tab1:
        st.subheader("Process an Image")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

        if uploaded_image is not None:
            # Read the image file
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Process image
            result, pipeline_images = lane_detector.process_image(image)

            # Display results
            st.subheader("Input Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

            st.subheader("Detected Lane")
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)

            # Show pipeline steps if requested
            if st.checkbox("Show pipeline steps"):
                display_pipeline_steps(pipeline_images)

    with tab2:
       st.subheader("Live Lane Detection on Video (Frame by Frame)")
       uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"], key="video_uploader")

       if uploaded_video is not None:
          # Save uploaded video to a temporary file
          with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
              tmp_file.write(uploaded_video.read())
              video_path = tmp_file.name

          # Initialize video capture and lane detector
          cap = cv2.VideoCapture(video_path)
          lane_detector = LaneDetector()

          # Streamlit video frame container
          stframe = st.empty()
          st.info("Click ▶ 'Play Video' to start processing frame-by-frame")

          # Buttons to control playback
          col1, col2 = st.columns([1, 1])
          play_btn = col1.button("▶ Play Video")
          stop_btn = col2.button("⏹ Stop Video")

          frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          fps = cap.get(cv2.CAP_PROP_FPS)
          st.caption(f"Total Frames: {frame_count} | FPS: {fps:.2f}")

          frame_number = 0

          while play_btn and cap.isOpened():
              ret, frame = cap.read()
              if not ret:
                  st.warning("End of video reached or failed to read frame.")
                  break

              try:
                  result_frame, _ = lane_detector.process_image(frame)
                  result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                  stframe.image(result_rgb, channels="RGB", use_column_width=True, caption=f"Frame {frame_number + 1}")
              except Exception as e:
                  st.warning(f"Error processing frame {frame_number + 1}: {e}")
                  break

              frame_number += 1

              # Break the loop if the user clicks Stop
              if stop_btn:
                  st.info("Video playback stopped.")
                  break

          cap.release()



if __name__ == '__main__':
    main()