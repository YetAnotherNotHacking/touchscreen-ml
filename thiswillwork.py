import cv2
import numpy as np
import mediapipe as mp
from pupil_apriltags import Detector
import cairosvg
import io
from collections import deque

class ProjectionTracker:
    def __init__(self):
        self.cap = self.setup_camera()
        self.detector = Detector(families='tag36h11')
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.5)
        self.projection_corners = None
        self.homography = None
        self.lines = deque(maxlen=2)  # Keep only the two most recent lines
        self.tag_positions = [(0, 0)] * 4  # Initialize with default positions
        self.blocked_tags = set()
        self.finger_positions = []
        self.tag_images = self.load_tag_images()
        self.window_name = 'Projection'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.tags_detected = 0
        self.total_lines_drawn = 0

    def setup_camera(self):
        # Prefer external camera (index 0) if available
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        return cap

    def load_tag_images(self):
        tag_images = []
        for i in range(4):
            svg_path = f"design/apriltags/tag{i}.svg"
            png_data = cairosvg.svg2png(url=svg_path)
            nparr = np.frombuffer(png_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            tag_images.append(cv2.resize(img, (100, 100)))  # Resize for display
        return tag_images

    def detect_apriltags(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray)
        corners = []
        for r in results:
            corners.append(r.corners)
        return corners

    def estimate_projection(self, corners):
        if len(corners) < 4:
            return None
        src_pts = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        dst_pts = np.float32(corners[:4])
        self.homography, _ = cv2.findHomography(src_pts, dst_pts)
        return dst_pts

    def detect_hands(self, frame):
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks
        return []

    def project_point(self, point):
        if self.homography is None:
            return None
        p = np.array([point[0], point[1], 1])
        projected = np.dot(self.homography, p)
        return (projected[0] / projected[2], projected[1] / projected[2])

    def reposition_tags(self, width, height):
        default_positions = [
            (0, 0),                # Top-left
            (width - 1, 0),        # Top-right
            (width - 1, height - 1),  # Bottom-right
            (0, height - 1)        # Bottom-left
        ]
        
        for i in range(4):
            if i in self.blocked_tags:
                # If blocked, move slightly inward
                x, y = default_positions[i]
                offset = 50  # pixels to move inward
                if i == 0:
                    x, y = x + offset, y + offset
                elif i == 1:
                    x, y = x - offset, y + offset
                elif i == 2:
                    x, y = x - offset, y - offset
                else:
                    x, y = x + offset, y - offset
                self.tag_positions[i] = (x, y)
            else:
                self.tag_positions[i] = default_positions[i]

    def process_frame(self, frame):
        corners = self.detect_apriltags(frame)
        self.tags_detected = len(corners)
        
        if corners:
            for i, corner in enumerate(corners):
                if i < 4:
                    self.tag_positions[i] = corner[0]
        
        if len(corners) < 4:
            self.blocked_tags = set(range(4)) - set(range(len(corners)))
            self.reposition_tags(frame.shape[1], frame.shape[0])

        self.projection_corners = self.estimate_projection(self.tag_positions)

        self.finger_positions = []
        if self.projection_corners is not None:
            hand_landmarks = self.detect_hands(frame)
            for hand_lms in hand_landmarks:
                index_finger_tip = hand_lms.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = index_finger_tip.x, index_finger_tip.y
                projected_point = self.project_point((x, y))
                if projected_point:
                    self.finger_positions.append(projected_point)
        
        # Update lines
        if self.finger_positions:
            self.lines.append(self.finger_positions)
            self.total_lines_drawn += 1

        return frame

    def draw_projection(self):
        window_size = cv2.getWindowImageRect(self.window_name)
        if window_size is None:
            return None
        _, _, width, height = window_size
        projection = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Reposition tags if necessary
        self.reposition_tags(width, height)
        
        # Draw AprilTags
        for i, pos in enumerate(self.tag_positions):
            if i < len(self.tag_images):
                tag_img = cv2.resize(self.tag_images[i], (100, 100))
                x, y = int(pos[0]), int(pos[1])
                
                # Ensure the tag is within the projection bounds
                x1, y1 = max(0, x - 50), max(0, y - 50)
                x2, y2 = min(width, x + 50), min(height, y + 50)
                
                # Calculate the portion of the tag image to use
                tag_x1, tag_y1 = max(0, 50 - x), max(0, 50 - y)
                tag_x2, tag_y2 = min(100, tag_x1 + (x2 - x1)), min(100, tag_y1 + (y2 - y1))
                
                # Place the visible portion of the tag on the projection
                projection[y1:y2, x1:x2] = tag_img[tag_y1:tag_y2, tag_x1:tag_x2]

        # Draw projection area
        if self.projection_corners is not None:
            scaled_corners = [(int(x * width), int(y * height)) for x, y in self.projection_corners]
            cv2.polylines(projection, [np.array(scaled_corners)], True, (0, 255, 0), 2)

        # Draw finger trails
        for line in self.lines:
            for i in range(len(line) - 1):
                start = (int(line[i][0] * width), int(line[i][1] * height))
                end = (int(line[i+1][0] * width), int(line[i+1][1] * height))
                cv2.line(projection, start, end, (0, 0, 255), 2)

        # Draw current finger positions
        for pos in self.finger_positions:
            x, y = int(pos[0] * width), int(pos[1] * height)
            cv2.circle(projection, (x, y), 10, (255, 0, 0), -1)

        # Add information text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(projection, f"Tags Detected: {self.tags_detected}", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(projection, f"Total Lines Drawn: {self.total_lines_drawn}", (10, 70), font, 1, (255, 255, 255), 2)
        cv2.putText(projection, f"Active Lines: {len(self.lines)}", (10, 110), font, 1, (255, 255, 255), 2)

        return projection

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            projection = self.draw_projection()

            if projection is not None:
                cv2.imshow('Camera Feed', processed_frame)
                cv2.imshow(self.window_name, projection)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ProjectionTracker()
    tracker.run()