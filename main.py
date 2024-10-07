# Real time tracking imports
import cv2
import mediapipe as mp

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import threading





# List of all mediapipe models
# FaceMesh
# Hands
# Holistic
# Objectron
# Pose
# SelfieSegmentation

def draw_line(start, end):
    glBegin(GL_LINES)
    glVertex3fv(start)
    glVertex3fv(end)
    glEnd()

# Function to draw the hand wireframe and detect fingertip touch
def draw_hand_wireframe(image, hand_landmarks):
    # Define connections between hand landmarks to draw skeleton
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Little finger
    ]

    # Draw circles at each hand landmark (joint)
    for landmark in hand_landmarks:
        cv2.circle(image, landmark, 5, (255, 255, 0), -1)
        # Add a label with the landmark index
        cv2.putText(image, str(hand_landmarks.index(landmark)), (landmark[0], landmark[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # Put text at ID 0
        if hand_landmarks.index(landmark) == 0:
            cv2.putText(image, "Wrist Base", (landmark[0], landmark[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 120), 1, cv2.LINE_AA)

    # Draw lines between connected hand landmarks
    for connection in connections:
        start_point = hand_landmarks[connection[0]]
        end_point = hand_landmarks[connection[1]]
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)

# Function to check if the fingertip touches the thumb
def detect_fingertip_touch(hand_landmarks, threshold=40, finger="pointer"):
    # Define connections between hand landmarks to check fingertip touch, meaning 4 and 8 touching
    if finger == "pointer":
        connections = [(4, 8)]
    elif finger == "middle":
        connections = [(4, 12)]
    elif finger == "ring":
        connections = [(4, 16)]
    elif finger == "little":
        connections = [(4, 20)]

    
    # Check if the fingertip touches the thumb
    for connection in connections:
        start_point = hand_landmarks[connection[0]]
        end_point = hand_landmarks[connection[1]]
        distance = ((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2) ** 0.5
        if distance < threshold:
            return True
    return False

# Function to draw body wireframe over image, not applied yet, done in process video
def draw_body_wireframe(image, landmarks, debug=False):
    # Define connections between body landmarks to draw skeleton
    connections = [
        (11, 12), (11, 23),  # Left arm
        (12, 24), (23, 24), (14, 16),  # Right arm
        (11, 13), (12, 14), (13, 15),  # Left forearm
        (23, 25), (24, 26),  # Right forearm
        (11, 23), (12, 24),  # Shoulders
        (11, 24), (12, 23),  # Neck
        (23, 24),  # Chest
        (11, 25), (12, 26),  # Hips
        (25, 27), (26, 28),  # Left thigh
        (27, 29), (28, 30),  # Left leg
        (29, 31), (30, 32)  # Left foot
    ]

    # Put a dot on each landmark except for 1 through 10
    for i, landmark in enumerate(landmarks):
        if i > 10:
            cv2.circle(image, landmark, 5, (255, 0, 0), -1)
            if debug == True:
                cv2.putText(image, str(i), (landmark[0], landmark[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 3, cv2.LINE_AA)




    # Draw lines between connected body landmarks
    for connection in connections:
        start_point = landmarks[connection[0]]
        end_point = landmarks[connection[1]]
        cv2.line(image, start_point, end_point, (255, 0, 255), 2)

# Function to draw face mesh
def draw_face_mesh(image, face_landmarks, debug=False, clown=True):
    # Draw connections
    connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
    for connection in connections:
        for i in range(len(connection) - 1):
            start_point = face_landmarks.landmark[connection[i]]
            end_point = face_landmarks.landmark[connection[i + 1]]
            start_point_px = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
            end_point_px = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))
            cv2.line(image, start_point_px, end_point_px, (0, 0, 255), 1)
            # Add dot to each point
            cv2.circle(image, start_point_px, 1, (0, 255, 255), -1)
            cv2.circle(image, end_point_px, 1, (0, 255, 255), -1)
            if debug == True:
                cv2.putText(image, str(connection[i]), start_point_px, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(connection[i + 1]), end_point_px, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Put dot on point 4 that is red
            if clown == True:
                if connection[i] == 4:
                    cv2.circle(image, start_point_px, 50, (0, 0, 255), -1)
                if connection[i + 1] == 4:
                    pass # Might make function here later


def render_human_model(body_landmarks):
    body_landmark_positions = body_landmarks
    # Initialize Pygame and create a window
    pygame.init()
    screen = pygame.display.set_mode((800, 600), DOUBLEBUF|OPENGL)

    # Set the perspective for the 3D scene
    gluPerspective(45, (800/600), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        """ Body Landmarks
        0. nose
        1. left_eye_inner 
        2. left_eye
        3. left_eye_outer
        4. right_eye_inner
        5. right_eye 
        6. right_eye_outer
        7. left_ear
        8. right_ear 
        9. mouth_left 
        10. mouth_right 
        11. left_shoulder 
        12. right_shoulder 
        13. left elbow
        14. right elbow
        15. left_wrist 
        16. right_wrist
        17. left pinky 
        18. right pinky 
        19. left_index
        20. right_index
        21. left_thumb 
        22. right_thumb
        23. left_hip 
        24. right_hip
        25. left knee
        26. right knee 
        27. left ankle 
        28. right ankle 
        29. left_heel 
        30. right_heel
        31. left_foot_index
        32. right_foot_index
        """

        # Draw the body parts
        draw_line(body_landmark_positions[0], body_landmark_positions[1])  # Head to neck
        draw_line(body_landmark_positions[1], body_landmark_positions[2])  # Neck to right shoulder
        draw_line(body_landmark_positions[1], body_landmark_positions[5])  # Neck to left shoulder
        draw_line(body_landmark_positions[2], body_landmark_positions[3])  # Right shoulder to right elbow
        draw_line(body_landmark_positions[3], body_landmark_positions[4])  # Right elbow to right wrist
        draw_line(body_landmark_positions[5], body_landmark_positions[6])  # Left shoulder to left elbow
        draw_line(body_landmark_positions[6], body_landmark_positions[7])  # Left elbow to left wrist
        draw_line(body_landmark_positions[1], body_landmark_positions[8])  # Neck to right hip
        draw_line(body_landmark_positions[1], body_landmark_positions[11])  # Neck to left hip
        draw_line(body_landmark_positions[8], body_landmark_positions[9])  # Right hip to right knee
        draw_line(body_landmark_positions[9], body_landmark_positions[10])  # Right knee to right ankle
        draw_line(body_landmark_positions[11], body_landmark_positions[12])  # Left hip to left knee
        draw_line(body_landmark_positions[12], body_landmark_positions[13])  # Left knee to left ankle

        pygame.display.flip()
        pygame.time.wait(10)









def process_video():
    # Initialize MediaPipe Hands, Pose, and FaceMesh models
    mp_hands = mp.solutions.hands.Hands()
    mp_pose = mp.solutions.pose.Pose()
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

    # Open default camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands, Pose, and FaceMesh
        results_hands = mp_hands.process(frame_rgb)
        results_pose = mp_pose.process(frame_rgb)
        results_face_mesh = mp_face_mesh.process(frame_rgb)

        



        # Draw hand wireframe
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                landmark_px = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in hand_landmarks.landmark]
                draw_hand_wireframe(frame, landmark_px)

                # Commented this out, the text is large an distracting, will add a circle instead that is smaller and a local text to the event

                # Check if fingertip touches the thumb
                # if detect_fingertip_touch(landmark_px, finger="pointer"):
                #     cv2.putText(frame, "Fingertip pointer", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # if detect_fingertip_touch(landmark_px, finger="middle"):
                #     cv2.putText(frame, "Fingertip middle", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # if detect_fingertip_touch(landmark_px, finger="ring"):
                #     cv2.putText(frame, "Fingertip ring", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # if detect_fingertip_touch(landmark_px, finger="little"):
                #     cv2.putText(frame, "Fingertip little", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if detect_fingertip_touch(landmark_px, finger="pointer"):
                    cv2.circle(frame, landmark_px[4], 25, (150, 25, 0), -1)
                    # Add text the the left/right of the circle
                    cv2.putText(frame, "Touch pointer", (landmark_px[4][0] + 30, landmark_px[4][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                if detect_fingertip_touch(landmark_px, finger="middle"):
                    cv2.circle(frame, landmark_px[12], 25, (100, 255, 100), -1)
                    # Add text the the left/right of the circle
                    cv2.putText(frame, "Touch middle", (landmark_px[12][0] + 30, landmark_px[12][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                if detect_fingertip_touch(landmark_px, finger="ring"):
                    cv2.circle(frame, landmark_px[16], 25, (100, 100, 255), -1)
                    # Add text the the left/right of the circle
                    cv2.putText(frame, "Touch ring", (landmark_px[16][0] + 30, landmark_px[16][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                if detect_fingertip_touch(landmark_px, finger="little"):
                    cv2.circle(frame, landmark_px[20], 25, (255, 25, 100), -1)
                    # Add text the the left/right of the circle
                    cv2.putText(frame, "Touch little", (landmark_px[20][0] + 30, landmark_px[20][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)




        # Draw face mesh
        if results_face_mesh.multi_face_landmarks:


            
            for face_landmarks in results_face_mesh.multi_face_landmarks:
                landmark_px = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in face_landmarks.landmark]
                draw_face_mesh(frame, face_landmarks)

        # Make new window showing the human model
        if results_pose.pose_landmarks:
            body_landmarks = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in results_pose.pose_landmarks.landmark]
            draw_body_wireframe(frame, body_landmarks)


        # Make neck
        # Connect point 11 from the body to 152 on the face mesh
        if results_pose.pose_landmarks and results_face_mesh.multi_face_landmarks:
            body_landmarks = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in results_pose.pose_landmarks.landmark]
            face_landmarks = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in results_face_mesh.multi_face_landmarks[0].landmark]
            # render_human_model_thread(body_landmarks)
            # Create a new process for the second window
            cv2.line(frame, body_landmarks[11], face_landmarks[377], (100, 25, 255), 2)
        # Connect point 12 from the body to 152 on the face mesh
            cv2.line(frame, body_landmarks[12], face_landmarks[148], (100, 25, 255), 2)
        # Connnect point 12 from the body to 149 on face mesh
            cv2.line(frame, body_landmarks[12], face_landmarks[149], (100, 25, 255), 2)
        # Connect point 11 from the body to 149 on face mesh
            cv2.line(frame, body_landmarks[11], face_landmarks[148], (100, 25, 255), 2)
        # Connect point 11 from the body to 400 on face mesh
            cv2.line(frame, body_landmarks[11], face_landmarks[400], (100, 25, 255), 2)
        # Connect point 12 from the body to 377 on face mesh
            cv2.line(frame, body_landmarks[12], face_landmarks[377], (100, 25, 255), 2)
        # Connect point 11 from the body 10 378 on face mesh
            cv2.line(frame, body_landmarks[11], face_landmarks[378], (100, 25, 255), 2)

            cv2.line(frame, body_landmarks[12], face_landmarks[377], (100, 25, 255), 2)
            cv2.line(frame, body_landmarks[12], face_landmarks[176], (100, 25, 255), 2)

        # Display the resulting frame
        cv2.imshow('Tracking Test 1', frame)





        # Check for exit key (press 'q' to exit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()