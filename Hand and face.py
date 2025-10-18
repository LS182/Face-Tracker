pip install opencv-python mediapipe


import cv2
import mediapipe as mp
import time
from collections import deque

# ---------- Configuration ----------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DISPLAY_LANDMARK_INDEXES = False  # set True to show landmark index numbers (looks noisy)
SMOOTH_LANDMARKS = True
# -----------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

# Drawing specs
hand_drawing_spec = mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=3)
hand_connection_spec = mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2)
face_drawing_spec = mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1)

# Utility: finger counting for a single hand
# Uses landmark indices from MediaPipe hands.
FINGER_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

def count_fingers(hand_landmarks, handedness_str):
    """
    Returns how many fingers are up (0-5) for one hand.
    This is a simple heuristic:
     - For fingers except thumb: tip_y < pip_y  == finger up (normalized coordinates)
     - For thumb: check x coordinate relative to index MCP (works if hand is fairly front-facing)
    handedness_str: 'Left' or 'Right' from MediaPipe classification
    """
    if hand_landmarks is None:
        return 0
    lm = hand_landmarks.landmark
    fingers_up = 0

    # For thumb: compare tip x to ip or mcp depending on left/right
    # Thumb is tricky; this approach works decently for many frontal views.
    if handedness_str == "Right":
        # For right hand, thumb is to the left for an open hand (x smaller)
        if lm[FINGER_TIPS[0]].x < lm[2].x:
            fingers_up += 1
    else:
        # For left hand, thumb is to the right for open hand (x larger)
        if lm[FINGER_TIPS[0]].x > lm[2].x:
            fingers_up += 1

    # For other fingers: tip y < pip y -> finger up (y=0 at top)
    finger_pip_ids = [6, 10, 14, 18]
    for tip_id, pip_id in zip(FINGER_TIPS[1:], finger_pip_ids):
        if lm[tip_id].y < lm[pip_id].y:
            fingers_up += 1

    return fingers_up

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # MediaPipe solutions
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5)

    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5)

    p_time = 0
    screenshot_count = 0

    # Small deque for smoothing FPS display if desired
    fps_deque = deque(maxlen=10)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (camera disconnected?). Exiting...")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            # ---------- Face Mesh ----------
            face_results = face_mesh.process(frame_rgb)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=face_drawing_spec)
                    # optional: draw small circle on each landmark
                    if DISPLAY_LANDMARK_INDEXES:
                        for idx, lm in enumerate(face_landmarks.landmark):
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)

            # ---------- Hands ----------
            hand_results = hands.process(frame_rgb)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, hand_handedness in zip(hand_results.multi_hand_landmarks,
                                                           hand_results.multi_handedness):
                    # handedness classification label: 'Left' or 'Right'
                    handedness_label = hand_handedness.classification[0].label

                    # draw landmarks + connections
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        hand_drawing_spec, hand_connection_spec)

                    # bounding box (tight)
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    # convert to pixel coords
                    x1, y1 = int(min_x * w) - 10, int(min_y * h) - 10
                    x2, y2 = int(max_x * w) + 10, int(max_y * h) + 10
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 50), 2)

                    # finger count
                    fingers_up = count_fingers(hand_landmarks, handedness_label)
                    label = f"{handedness_label} hand: {fingers_up} fingers"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 50), 2)

                    # optionally draw landmark indices
                    if DISPLAY_LANDMARK_INDEXES:
                        for i, lm in enumerate(hand_landmarks.landmark):
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.putText(frame, str(i), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)

            # ---------- FPS ----------
            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time
            fps_deque.append(fps)
            fps_display = sum(fps_deque) / len(fps_deque)
            cv2.putText(frame, f'FPS: {int(fps_display)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save screenshot", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

            cv2.imshow('Face + Hand Tracker', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()

if __name__ == "__main__":
    main()
