import cv2
import numpy as np
from pupil_apriltags import Detector
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import threading
import io
import cairosvg
from collections import deque

# Initialize AprilTag detector
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Initialize webcam
cap_index = 0
cap = cv2.VideoCapture(cap_index)

# Global variables
running = True
tag_positions = {}
board_corners = None
hand_landmarks = None
apriltag_images = {}
canvas_width = 800
canvas_height = 600
virtual_board = np.zeros((600, 800, 3), dtype=np.uint8)
last_hand_positions = deque(maxlen=100)

# Function to cycle through available cameras
def switch_camera(event=None):
    global cap, cap_index
    cap_index += 1
    if not cap.open(cap_index):
        cap_index = 0
        cap.open(cap_index)

# Function to detect hand landmarks using MediaPipe
def detect_hand(frame):
    global hand_landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    hand_landmarks = result.multi_hand_landmarks

# Function to draw hand landmarks on the frame
def draw_hand_landmarks(frame, hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            for landmark in landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

# Function to detect board corners
def detect_board(tag_positions):
    if len(tag_positions) < 2:
        return None
    
    # Define the expected 3D positions of the tags (assuming a rectangular board)
    tag_3d_positions = {
        0: (0, 0, 0),
        1: (1, 0, 0),
        2: (1, 1, 0),
        3: (0, 1, 0)
    }
    
    # Collect the detected tag positions
    points_3d = []
    points_2d = []
    for tag_id, center in tag_positions.items():
        points_3d.append(tag_3d_positions[tag_id])
        points_2d.append(center)
    
    # Convert to numpy arrays
    points_3d = np.array(points_3d, dtype=np.float32)
    points_2d = np.array(points_2d, dtype=np.float32)
    
    # If we have less than 4 points, we can't use solvePnP
    if len(points_3d) < 4:
        # Use a simple estimation based on the available points
        if len(points_3d) == 2:
            # Estimate the other two corners
            vector = points_2d[1] - points_2d[0]
            perpendicular = np.array([-vector[1], vector[0]])
            estimated_corners = np.array([
                points_2d[0],
                points_2d[1],
                points_2d[1] + perpendicular,
                points_2d[0] + perpendicular
            ])
            return estimated_corners.astype(int)
        elif len(points_3d) == 3:
            # Use the three detected corners and estimate the fourth
            detected_corners = points_2d
            center = np.mean(detected_corners, axis=0)
            fourth_corner = 2 * center - detected_corners[0]
            estimated_corners = np.vstack((detected_corners, fourth_corner))
            return estimated_corners.astype(int)
        
    fx = 800  # Focal length in pixels
    fy = 600
    cx = -400  # Principal point (center of image)
    cy = -300
    
    # If we have 4 or more points, use solvePnP
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion
    
    try:
        success, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs)
        
        if not success:
            return None
        
        # Project all four corners
        all_corners_3d = np.array(list(tag_3d_positions.values()), dtype=np.float32)
        projected_corners, _ = cv2.projectPoints(all_corners_3d, rvec, tvec, camera_matrix, dist_coeffs)
        
        return projected_corners.reshape(-1, 2).astype(int)
    except cv2.error:
        # If solvePnP fails, fall back to using the detected points directly
        return points_2d.astype(int)

# Function to check if hand is inside the board and get its position
def get_hand_position_on_board(hand_landmarks, board_corners):
    if not hand_landmarks or board_corners is None:
        return None
    
    for landmarks in hand_landmarks:
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = index_tip.x, index_tip.y
        
        if cv2.pointPolygonTest(board_corners, (x * 640, y * 480), False) >= 0:
            # Convert hand position to board coordinates
            src_pts = np.array(board_corners, dtype=np.float32)
            dst_pts = np.array([[0, 0], [800, 0], [800, 600], [0, 600]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            pos = cv2.perspectiveTransform(np.array([[[x * 640, y * 480]]], dtype=np.float32), matrix)
            return (int(pos[0][0][0]), int(pos[0][0][1]))
    return None

# Camera thread function for capturing AprilTags and hand landmarks
def camera_thread():
    global running, tag_positions, board_corners, hand_landmarks
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)

        tag_positions.clear()
        for tag in tags:
            tag_id = tag.tag_id
            center = tuple(map(int, tag.center))
            tag_positions[tag_id] = center

        # Detect board
        board_corners = detect_board(tag_positions)

        # Detect hand landmarks
        detect_hand(frame)

        # Draw AprilTag centers and board bounding box
        for center in tag_positions.values():
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        if board_corners is not None:
            cv2.drawContours(frame, [board_corners], 0, (0, 255, 0), 2)

        # Draw hand landmarks
        draw_hand_landmarks(frame, hand_landmarks)

        # Convert to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update camera label
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

    cap.release()

# Function to load AprilTags (as SVG) into the interface
def load_apriltags():
    tag_ids = [0, 1, 2, 3]
    for tag_id in tag_ids:
        svg_path = f'design/apriltags/tag{tag_id}.svg'  # Assuming the images are named tag0.svg, tag1.svg, etc.
        
        # Convert SVG to PNG in memory
        png_data = cairosvg.svg2png(url=svg_path)
        image = Image.open(io.BytesIO(png_data))
        
        # Resize if necessary
        image = image.resize((80, 80), Image.LANCZOS)  # Resizing to fit the display
        apriltag_images[tag_id] = ImageTk.PhotoImage(image)

# Create main Tkinter window
root = tk.Tk()
root.title("AprilTag & Hand Detection with Drawing")

# Make the window resizable
root.geometry(f"{canvas_width}x{canvas_height}")
root.resizable(True, True)

# Bind space key for camera switching
root.bind("<space>", switch_camera)

# Create canvas for displaying interactive area
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack(fill=tk.BOTH, expand=True)

# Create labels for AprilTags
tag_labels = {
    "top_left": tk.Label(root, text="Top Left (ID 0): Missing", fg="red"),
    "top_right": tk.Label(root, text="Top Right (ID 1): Missing", fg="red"),
    "bottom_right": tk.Label(root, text="Bottom Right (ID 2): Missing", fg="red"),
    "bottom_left": tk.Label(root, text="Bottom Left (ID 3): Missing", fg="red")
}

for label in tag_labels.values():
    label.pack()

# Create a separate window for camera feed
camera_window = tk.Toplevel(root)
camera_window.title("Camera Feed")
camera_label = tk.Label(camera_window)
camera_label.pack()

# Function to draw AprilTag images on canvas
def draw_apriltag(canvas, x, y, size, tag_id, detected):
    image = apriltag_images[tag_id]
    canvas.create_image(x, y, anchor=tk.NW, image=image)
    canvas.create_text(x + size // 2, y + size + 5, text=f"ID {tag_id}", font=("Arial", 8))
    canvas.create_text(x + size // 2, y + size + 20, text="Detected" if detected else "Missing", font=("Arial", 8), fill="green" if detected else "red")

# Function to handle window resizing
def on_resize(event):
    global canvas_width, canvas_height
    canvas_width = event.width
    canvas_height = event.height
    canvas.config(width=canvas_width, height=canvas_height)

# Bind the resize event
root.bind("<Configure>", on_resize)

# Function to draw on the virtual board
def draw_on_virtual_board(pos):
    global virtual_board, last_hand_positions
    last_hand_positions.append(pos)
    for p in last_hand_positions:
        cv2.circle(virtual_board, p, 2, (0, 0, 255), -1)

# Function to clear the virtual board
def clear_virtual_board():
    global virtual_board, last_hand_positions
    virtual_board = np.zeros((600, 800, 3), dtype=np.uint8)
    last_hand_positions.clear()

# Create clear button
clear_button = tk.Button(root, text="Clear Board", command=clear_virtual_board)
clear_button.pack()

# Update the GUI display
def update_gui():
    # Clear previous drawings
    canvas.delete("all")

    # Draw virtual board
    board_image = Image.fromarray(virtual_board)
    board_image = board_image.resize((canvas_width, canvas_height), Image.LANCZOS)
    board_photo = ImageTk.PhotoImage(board_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=board_photo)
    canvas.image = board_photo  # Keep a reference to prevent garbage collection

    # Draw AprilTags in corners (detected or missing)
    tag_size = 80
    draw_apriltag(canvas, 10, 10, tag_size, 0, 0 in tag_positions)  # Top Left
    draw_apriltag(canvas, canvas_width - tag_size - 10, 10, tag_size, 1, 1 in tag_positions)  # Top Right
    draw_apriltag(canvas, canvas_width - tag_size - 10, canvas_height - tag_size - 10, tag_size, 2, 2 in tag_positions)  # Bottom Right
    draw_apriltag(canvas, 10, canvas_height - tag_size - 10, tag_size, 3, 3 in tag_positions)  # Bottom Left

    # Draw board corners if detected
    if board_corners is not None:
        scaled_corners = [(int(x * canvas_width / 640), int(y * canvas_height / 480)) for x, y in board_corners]
        canvas.create_polygon(scaled_corners, outline='green', fill='', width=2)

    # Check if hand is inside the board and draw
    if hand_landmarks and board_corners is not None:
        finger_pos = get_hand_position_on_board(hand_landmarks, board_corners)
        if finger_pos is not None:
            canvas.create_text(canvas_width // 2, 30, text="Drawing on board", font=("Arial", 16), fill="green")
            # Scale finger position to canvas size
            x = int(finger_pos[0] * canvas_width / 800)
            y = int(finger_pos[1] * canvas_height / 600)
            canvas.create_oval(x-5, y-5, x+10, y+10, fill='red')
            draw_on_virtual_board(finger_pos)
        else:
            canvas.create_text(canvas_width // 2, 30, text="Hand outside board", font=("Arial", 16), fill="red")

    # Update tag labels
    for i, position in enumerate(["top_left", "top_right", "bottom_right", "bottom_left"]):
        detected = i in tag_positions
        tag_labels[position].config(
            text=f"{position.replace('_', ' ').title()} (ID {i}): {'Detected' if detected else 'Missing'}",
            fg="green" if detected else "red"
        )

    root.after(50, update_gui)

# Start the camera thread
camera_thread = threading.Thread(target=camera_thread)
camera_thread.daemon = True
camera_thread.start()

# Load AprilTag images
load_apriltags()

# Start updating the GUI
update_gui()

# Start Tkinter main loop
root.mainloop()

# Cleanup
running = False
camera_thread.join()
hands.close()