
# import streamlit as st
# import cv2
# import numpy as np
# import tempfile
# from PIL import Image

# st.set_page_config(layout="wide")
# st.title("🛣️ Traditional Lane Detection (Canny + Hough)")

# uploaded_file = st.file_uploader("Upload Road Image or Video", type=["jpg", "jpeg", "png", "mp4"])

# threshold1 = st.slider("🔧 Canny Threshold 1", 0, 255, 50)
# threshold2 = st.slider("🔧 Canny Threshold 2", 0, 255, 150)
# hough_thresh = st.slider("🔧 Hough Threshold", 10, 200, 80)
# min_line_len = st.slider("📏 Min Line Length", 10, 200, 50)
# max_line_gap = st.slider("📏 Max Line Gap", 10, 200, 50)
# overlay = st.checkbox("🌓 Show Overlay Instead of Side-by-Side", value=False)

# def detect_lanes(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, threshold1, threshold2)
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_thresh, minLineLength=min_line_len, maxLineGap=max_line_gap)
#     lane_img = img.copy()
#     if lines is not None:
#         for line in lines:
#             x1,y1,x2,y2 = line[0]
#             cv2.line(lane_img, (x1,y1), (x2,y2), (0,255,0), 2)
#     return edges, lane_img

# if uploaded_file:
#     if uploaded_file.type.startswith("video"):
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())
#         cap = cv2.VideoCapture(tfile.name)
#         ret, frame = cap.read()
#         if ret:
#             edges, lane_img = detect_lanes(frame)
#             st.subheader("🎥 First Frame of Uploaded Video")
#             col1, col2 = st.columns(2)
#             col1.image(frame, caption="Original Frame", channels="BGR")
#             col2.image(lane_img, caption="Detected Lanes", channels="BGR")
#     else:
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, 1)
#         edges, lane_img = detect_lanes(img)
#         if overlay:
#             blended = cv2.addWeighted(img, 0.7, lane_img, 0.3, 0)
#             st.image(blended, caption="Overlay: Original + Detected Lanes", channels="BGR", use_container_width=True)
#         else:
#             col1, col2 = st.columns(2)
#             col1.image(img, caption="Original", channels="BGR", use_container_width=True)
#             col2.image(lane_img, caption="Detected Lanes", channels="BGR", use_container_width=True)


#  Best Code

# import streamlit as st
# import cv2
# import numpy as np
# import tempfile

# st.set_page_config(layout="wide")
# st.title("🛣️ Video Lane Detection")

# uploaded_file = st.file_uploader("Upload Road Video", type=["mp4", "avi", "mov"])

# # Detection parameters
# threshold1 = st.slider("Canny Threshold 1", 50, 150, 50)
# threshold2 = st.slider("Canny Threshold 2", 100, 250, 150)
# hough_thresh = st.slider("Hough Threshold", 20, 100, 50)
# min_line_len = st.slider("Min Line Length", 20, 150, 50)
# max_line_gap = st.slider("Max Line Gap", 5, 50, 20)
# frame_skip = st.slider("Process Every N Frames", 1, 10, 3)

# def detect_lanes(frame):
#     # 1. Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # 2. Apply blur
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # 3. Canny edge detection
#     edges = cv2.Canny(blur, threshold1, threshold2)
    
#     # 4. Region of interest (lower half trapezoid)
#     height, width = edges.shape
#     mask = np.zeros_like(edges)
#     roi = np.array([[
#         (width*0.1, height),
#         (width*0.9, height),
#         (width*0.6, height*0.6),
#         (width*0.4, height*0.6)
#     ]], np.int32)
#     cv2.fillPoly(mask, roi, 255)
#     masked_edges = cv2.bitwise_and(edges, mask)
    
#     # 5. Hough line transform
#     lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, hough_thresh,
#                           minLineLength=min_line_len, maxLineGap=max_line_gap)
    
#     # 6. Draw lane lines
#     line_img = np.zeros_like(frame)
#     if lines is not None:
#         left_lines = []
#         right_lines = []
        
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             slope = (y2-y1)/(x2-x1+0.001)  # Avoid division by zero
            
#             # Classify as left or right lane
#             if abs(slope) > 0.5:  # Filter horizontal lines
#                 if slope < 0 and x1 < width/2 and x2 < width/2:
#                     left_lines.append(line[0])
#                 elif slope > 0 and x1 > width/2 and x2 > width/2:
#                     right_lines.append(line[0])
        
#         # Draw averaged left lane
#         if left_lines:
#             left_avg = np.average(left_lines, axis=0).astype(int)
#             cv2.line(line_img, (left_avg[0], left_avg[1]), 
#                     (left_avg[2], left_avg[3]), (0,255,0), 5)
        
#         # Draw averaged right lane
#         if right_lines:
#             right_avg = np.average(right_lines, axis=0).astype(int)
#             cv2.line(line_img, (right_avg[0], right_avg[1]), 
#                     (right_avg[2], right_avg[3]), (0,255,0), 5)
    
#     # 7. Combine with original frame
#     return cv2.addWeighted(frame, 0.8, line_img, 1, 0)

# if uploaded_file:
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded_file.read())
    
#     cap = cv2.VideoCapture(tfile.name)
#     stframe = st.empty()
    
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue
            
#         # Process frame
#         processed = detect_lanes(frame)
        
#         # Display in Streamlit
#         stframe.image(processed, channels="BGR", 
#                      caption=f"Processed Frame {frame_count}")
        
#         # Add a small delay to allow display to update
#         cv2.waitKey(10)
    
#     cap.release()
#     st.success("Video processing complete!")

# Another one(Frame by Frame)

# import streamlit as st
# import cv2
# import numpy as np
# import tempfile

# st.set_page_config(layout="wide")
# st.title("🚗 Advanced Video Lane Detection Dashboard")

# # Sidebar controls
# with st.sidebar:
#     st.header("Control Panel")
#     threshold1 = st.slider("Canny Threshold 1", 50, 150, 50, help="Lower threshold for edge detection")
#     threshold2 = st.slider("Canny Threshold 2", 100, 250, 150, help="Upper threshold for edge detection")
#     hough_thresh = st.slider("Hough Threshold", 20, 100, 50, help="Minimum votes for line detection")
#     min_line_len = st.slider("Min Line Length", 20, 150, 50, help="Minimum length of detected lines (pixels)")
#     max_line_gap = st.slider("Max Line Gap", 5, 50, 20, help="Maximum gap between line segments")
#     frame_skip = st.slider("Process Every N Frames", 1, 10, 3, help="Higher values improve performance")
#     show_roi = st.checkbox("Show ROI Mask", False, help="Display region of interest mask")
#     show_edges = st.checkbox("Show Edge Detection", False, help="Display Canny edge detection")

# def get_roi_mask(frame):
#     height, width = frame.shape[:2]
#     mask = np.zeros((height, width), dtype=np.uint8)
#     roi_corners = np.array([[
#         (int(width*0.1), height),
#         (int(width*0.9), height),
#         (int(width*0.6), int(height*0.6)),
#         (int(width*0.4), int(height*0.6))
#     ]], dtype=np.int32)
#     cv2.fillPoly(mask, roi_corners, 255)
#     return mask

# def detect_lanes(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, threshold1, threshold2)
    
#     # Apply ROI mask
#     roi_mask = get_roi_mask(frame)
#     masked_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    
#     # Detect lines
#     lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, hough_thresh,
#                           minLineLength=min_line_len, maxLineGap=max_line_gap)
    
#     # Create output frame
#     line_img = np.zeros_like(frame)
#     if lines is not None:
#         left_lines, right_lines = [], []
        
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             slope = (y2-y1)/(x2-x1+0.001)  # Avoid division by zero
            
#             if abs(slope) > 0.5:  # Filter horizontal lines
#                 if slope < 0: left_lines.append(line[0])
#                 else: right_lines.append(line[0])
        
#         # Draw averaged lanes
#         def draw_average_line(lines, color):
#             if lines:
#                 avg = np.average(lines, axis=0).astype(int)
#                 cv2.line(line_img, (avg[0], avg[1]), (avg[2], avg[3]), color, 5)
        
#         draw_average_line(left_lines, (0, 255, 0))  # Green for left lane
#         draw_average_line(right_lines, (0, 0, 255))  # Red for right lane
    
#     # Combine with original frame
#     result = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
    
#     # Prepare debug views if needed
#     debug_views = []
#     if show_edges:
#         debug_views.append(("Edges", cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)))
#     if show_roi:
#         roi_visual = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
#         cv2.polylines(roi_visual, [np.array([
#             [int(frame.shape[1]*0.1), frame.shape[0]],
#             [int(frame.shape[1]*0.9), frame.shape[0]],
#             [int(frame.shape[1]*0.6), int(frame.shape[0]*0.6)],
#             [int(frame.shape[1]*0.4), int(frame.shape[0]*0.6)]
#         ], dtype=np.int32)], True, (0,255,255), 2)
#         debug_views.append(("ROI", roi_visual))
    
#     return result, debug_views

# # Main interface
# uploaded_file = st.file_uploader("Upload Road Video", type=["mp4", "avi", "mov"])

# if uploaded_file:
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded_file.read())
    
#     cap = cv2.VideoCapture(tfile.name)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     st.info(f"Video loaded: {frame_count} frames @ {fps:.1f} FPS")
    
#     # Create display containers
#     main_col = st.columns(1)[0]
#     debug_cols = st.columns(2 if (show_edges or show_roi) else 1)
    
#     frame_counter = 0
#     stop_processing = st.button("Stop Processing")
    
#     while cap.isOpened() and not stop_processing:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame_counter += 1
#         if frame_counter % frame_skip != 0:
#             continue
            
#         # Process frame
#         processed, debug_views = detect_lanes(frame)
        
#         # Display main result
#         with main_col:
#             st.image(processed, channels="BGR", 
#                    caption=f"Frame {frame_counter}", use_column_width=True)
        
#         # Display debug views if enabled
#         if debug_views:
#             for i, (name, view) in enumerate(debug_views):
#                 with debug_cols[i]:
#                     st.image(view, caption=name, use_column_width=True)
    
#     cap.release()
#     st.success(f"Finished processing {frame_counter} frames!")
#     st.balloons()


# import streamlit as st
# import cv2
# import numpy as np
# import tempfile

# st.set_page_config(layout="wide")
# st.title("🛣️ Video Lane Detection")

# uploaded_file = st.file_uploader("Upload Road Video", type=["mp4", "avi", "mov"])

# # Default slider values
# default_values = {
#     "threshold1": 50,
#     "threshold2": 150,
#     "hough_thresh": 50,
#     "min_line_len": 50,
#     "max_line_gap": 20,
#     "frame_skip": 3,
#     "lane_color": "Green"
# }

# # Sidebar settings
# with st.sidebar:
#     st.header("⚙️ Detection Settings")

#     # Sliders
#     threshold1 = st.sidebar.slider("Canny Threshold 1", 50, 150, st.session_state.get("threshold1", 50))
#     threshold2 = st.sidebar.slider("Canny Threshold 2", 100, 250, st.session_state.get("threshold2", 150))
#     hough_thresh = st.sidebar.slider("Hough Threshold", 20, 100, st.session_state.get("hough_thresh", 50))
#     min_line_len = st.sidebar.slider("Min Line Length", 20, 150, st.session_state.get("min_line_len", 50))
#     max_line_gap = st.sidebar.slider("Max Line Gap", 5, 50, st.session_state.get("max_line_gap", 20))
#     frame_skip = st.sidebar.slider("Process Every N Frames", 1, 10, st.session_state.get("frame_skip", 3))


#     # Lane color picker
#     lane_color = st.radio("Lane Color", ["Green", "Red", "Blue", "Yellow"])

#     color_dict = {
#     "Green": (0, 255, 0),      # BGR
#     "Red": (0, 0, 255),        # Red in BGR
#     "Blue": (255, 0, 0),       # Blue in BGR
#     "Yellow": (0, 255, 255)    # Yellow in BGR
# }

# def detect_lanes(frame, color):
#     # 1. Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # 2. Apply blur
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     # 3. Canny edge detection
#     edges = cv2.Canny(blur, threshold1, threshold2)

#     # 4. ROI mask
#     height, width = edges.shape
#     mask = np.zeros_like(edges)
#     roi = np.array([[
#         (width*0.1, height),
#         (width*0.9, height),
#         (width*0.6, height*0.6),
#         (width*0.4, height*0.6)
#     ]], np.int32)
#     cv2.fillPoly(mask, roi, 255)
#     masked_edges = cv2.bitwise_and(edges, mask)

#     # 5. Hough transform
#     lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, hough_thresh,
#                             minLineLength=min_line_len, maxLineGap=max_line_gap)

#     # 6. Draw lane lines
#     line_img = np.zeros_like(frame)
#     if lines is not None:
#         left_lines, right_lines = [], []

#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             slope = (y2 - y1) / (x2 - x1 + 0.001)
#             if abs(slope) > 0.5:
#                 if slope < 0 and x1 < width/2 and x2 < width/2:
#                     left_lines.append(line[0])
#                 elif slope > 0 and x1 > width/2 and x2 > width/2:
#                     right_lines.append(line[0])

#         # Average and draw
#         if left_lines:
#             left_avg = np.average(left_lines, axis=0).astype(int)
#             cv2.line(line_img, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), color, 5)
#         if right_lines:
#             right_avg = np.average(right_lines, axis=0).astype(int)
#             cv2.line(line_img, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), color, 5)

#     return cv2.addWeighted(frame, 0.8, line_img, 1, 0)

# if uploaded_file:
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded_file.read())

#     cap = cv2.VideoCapture(tfile.name)
#     stframe = st.empty()
#     first_frame_displayed = False

#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue

#         processed = detect_lanes(frame, color=color_dict[lane_color])

#         # First frame preview in sidebar
#         if not first_frame_displayed:
#             with st.sidebar:
#                 st.markdown("### 🖼️ First Frame Preview")
#                 st.image(frame, channels="BGR", use_column_width=True)
#             first_frame_displayed = True

#         stframe.image(processed, channels="BGR", caption=f"Processed Frame {frame_count}")
#         cv2.waitKey(10)

#     cap.release()
#     st.success(" Video processing complete!")

# import streamlit as st
# import cv2
# import numpy as np
# import tempfile
# from PIL import Image

# st.set_page_config(layout="wide")
# st.title("🛣️ Image + Video Lane Detection")

# uploaded_file = st.file_uploader("Upload Road Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# # Default slider values
# default_values = {
#     "threshold1": 50,
#     "threshold2": 150,
#     "hough_thresh": 50,
#     "min_line_len": 50,
#     "max_line_gap": 20,
#     "frame_skip": 3
# }

# # Sidebar settings
# with st.sidebar:
#     st.header("⚙️ Detection Settings")
    
#     threshold1 = st.slider("Canny Threshold 1", 50, 150, default_values["threshold1"])
#     threshold2 = st.slider("Canny Threshold 2", 100, 250, default_values["threshold2"])
#     hough_thresh = st.slider("Hough Threshold", 20, 100, default_values["hough_thresh"])
#     min_line_len = st.slider("Min Line Length", 20, 150, default_values["min_line_len"])
#     max_line_gap = st.slider("Max Line Gap", 5, 50, default_values["max_line_gap"])
#     frame_skip = st.slider("Process Every N Frames", 1, 10, default_values["frame_skip"])
    
#     lane_color = st.radio("Lane Color", ["Green", "Red", "Blue", "Yellow"])

#     color_dict = {
#         "Green": (0, 255, 0),
#         "Red": (0, 0, 255),
#         "Blue": (255, 0, 0),
#         "Yellow": (0, 255, 255)
#     }

# def detect_lanes(frame, color):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, threshold1, threshold2)

#     height, width = edges.shape
#     mask = np.zeros_like(edges)
#     roi = np.array([[
#         (width*0.1, height),
#         (width*0.9, height),
#         (width*0.6, height*0.6),
#         (width*0.4, height*0.6)
#     ]], np.int32)
#     cv2.fillPoly(mask, roi, 255)
#     masked_edges = cv2.bitwise_and(edges, mask)

#     lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, hough_thresh,
#                             minLineLength=min_line_len, maxLineGap=max_line_gap)

#     line_img = np.zeros_like(frame)
#     if lines is not None:
#         left_lines, right_lines = [], []

#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             slope = (y2 - y1) / (x2 - x1 + 0.001)
#             if abs(slope) > 0.5:
#                 if slope < 0 and x1 < width/2 and x2 < width/2:
#                     left_lines.append(line[0])
#                 elif slope > 0 and x1 > width/2 and x2 > width/2:
#                     right_lines.append(line[0])

#         if left_lines:
#             left_avg = np.average(left_lines, axis=0).astype(int)
#             cv2.line(line_img, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), color, 5)
#         if right_lines:
#             right_avg = np.average(right_lines, axis=0).astype(int)
#             cv2.line(line_img, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), color, 5)

#     return cv2.addWeighted(frame, 0.8, line_img, 1, 0)

# if uploaded_file:
#     file_type = uploaded_file.type

#     if file_type.startswith("image"):
#         # Load and prepare image
#         image = np.array(Image.open(uploaded_file).convert("RGB"))
#         image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Clone for display and processing
#         original = image_bgr.copy()
#         processed = detect_lanes(image_bgr.copy(), color=color_dict[lane_color])

#         # Side-by-side display
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(original, caption="Original Image", channels="BGR", use_column_width=True)
#         with col2:
#             st.image(processed, caption="Detected Lanes", channels="BGR", use_column_width=True)

#     elif file_type.startswith("video"):
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())
#         cap = cv2.VideoCapture(tfile.name)
#         stframe = st.empty()
#         first_frame_displayed = False
#         frame_count = 0


#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_count += 1
#             if frame_count % frame_skip != 0:
#                 continue

#             processed = detect_lanes(frame, color=color_dict[lane_color])

#             if not first_frame_displayed:
#                 with st.sidebar:
#                     st.markdown("### 🖼️ First Frame Preview")
#                     st.image(frame, channels="BGR", use_column_width=True)
#                 first_frame_displayed = True

#             stframe.image(processed, channels="BGR", caption=f"Processed Frame {frame_count}")
#             cv2.waitKey(10)

#         cap.release()
#         st.success(" Video processing complete!")

import streamlit as st
import cv2
import numpy as np
import tempfile

st.set_page_config(layout="wide")
st.title("🛣️ Video Lane Detection")

uploaded_file = st.file_uploader("Upload Road Video", type=["mp4", "avi", "mov"])

# Default slider values
default_values = {
    "threshold1": 50,
    "threshold2": 150,
    "hough_thresh": 50,
    "min_line_len": 50,
    "max_line_gap": 20,
    "frame_skip": 3
}

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Detection Settings")
    
    threshold1 = st.slider("Canny Threshold 1", 50, 150, default_values["threshold1"])
    threshold2 = st.slider("Canny Threshold 2", 100, 250, default_values["threshold2"])
    hough_thresh = st.slider("Hough Threshold", 20, 100, default_values["hough_thresh"])
    min_line_len = st.slider("Min Line Length", 20, 150, default_values["min_line_len"])
    max_line_gap = st.slider("Max Line Gap", 5, 50, default_values["max_line_gap"])
    frame_skip = st.slider("Process Every N Frames", 1, 10, default_values["frame_skip"])
    
    lane_color = st.radio("Lane Color", ["Green", "Red", "Blue", "Yellow"])

    color_dict = {
        "Green": (0, 255, 0),
        "Red": (0, 0, 255),
        "Blue": (255, 0, 0),
        "Yellow": (0, 255, 255)
    }

# Lane detection logic
def detect_lanes(frame, color):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, threshold1, threshold2)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    roi = np.array([[  # Trapezoid ROI
        (width*0.1, height),
        (width*0.9, height),
        (width*0.6, height*0.6),
        (width*0.4, height*0.6)
    ]], np.int32)
    cv2.fillPoly(mask, roi, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, hough_thresh,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    line_img = np.zeros_like(frame)
    if lines is not None:
        left_lines, right_lines = [], []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(slope) > 0.5:
                if slope < 0 and x1 < width/2 and x2 < width/2:
                    left_lines.append(line[0])
                elif slope > 0 and x1 > width/2 and x2 > width/2:
                    right_lines.append(line[0])

        if left_lines:
            left_avg = np.mean(left_lines, axis=0).astype(int)
            cv2.line(line_img, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), color, 5)
        if right_lines:
            right_avg = np.mean(right_lines, axis=0).astype(int)
            cv2.line(line_img, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), color, 5)

    return cv2.addWeighted(frame, 0.8, line_img, 1, 0)

# Video handling
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    first_frame_displayed = False
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        processed = detect_lanes(frame, color=color_dict[lane_color])

        if not first_frame_displayed:
            with st.sidebar:
                st.markdown("### 🖼️ First Frame Preview")
                st.image(frame, channels="BGR", use_container_width=True)
            first_frame_displayed = True

        stframe.image(processed, channels="BGR", caption=f"Processed Frame {frame_count}")
        cv2.waitKey(10)

    cap.release()
    st.success(" Video processing complete!")
