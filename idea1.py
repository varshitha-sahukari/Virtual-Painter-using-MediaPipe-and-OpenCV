import mediapipe as mp
import cv2
import numpy as np
import time
import os
import uuid

# Constants
ml = 150
max_x, max_y = 250 + ml, 100  # Toolbar: 100px height, 250px width (400-150)
curr_tool = "select tool"
curr_color = (0, 0, 255)  # Default color: red
color_name = "red"
rad = 40
thick = 4
prevx, prevy = 0, 0
var_inits = False
time_init = True
tool_select_delay = 0.8

# Tool selection function


def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    else:
        return "erase"

# Color selection function


def getColor(x):
    if x < 50 + ml:
        return (0, 0, 255), "red"  # Red
    elif x < 100 + ml:
        return (0, 255, 0), "green"  # Green
    elif x < 150 + ml:
        return (255, 0, 0), "blue"  # Blue
    elif x < 200 + ml:
        return (0, 0, 0), "black"  # Black
    else:
        return (0, 255, 255), "yellow"  # Yellow

# Check if index finger is raised


def index_raised(yi, y9):
    return (y9 - yi) > 40

# Check if hand is open (for eraser)


def is_hand_open(landmarks):
    # Get coordinates of wrist (0) and fingertips (4, 8, 12, 16, 20)
    wrist = np.array([landmarks[0].x * 640, landmarks[0].y * 480])
    fingertips = [
        np.array([landmarks[i].x * 640, landmarks[i].y * 480])
        for i in [4, 8, 12, 16, 20]
    ]
    # Calculate distances from wrist to each fingertip
    distances = [np.linalg.norm(tip - wrist) for tip in fingertips]
    # Check if all distances are above a threshold (indicating open hand)
    threshold = 100  # Adjust based on testing
    return all(d > threshold for d in distances)


# Initialize MediaPipe Hands
hands = mp.solutions.hands
hand_landmark = hands.Hands(
    min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Create toolbar with shape and color buttons
tools = np.zeros((max_y, max_x - ml, 3), dtype="uint8")  # Shape: (100, 250, 3)
# Shape buttons (top row, y: 0-50)
cv2.rectangle(tools, (0, 0), (max_x - ml, 50),
              (255, 255, 255), -1)  # White background
cv2.rectangle(tools, (0, 0), (max_x - ml, 50), (0, 0, 255), 2)  # Red border
for x in [50, 100, 150, 200]:
    cv2.line(tools, (x, 0), (x, 50), (0, 0, 255), 2)
# Add text labels for shape buttons
tool_labels = ["Line", "Rect", "Draw", "Circle", "Erase"]
for i, label in enumerate(tool_labels):
    cv2.putText(tools, label, (i * 50 + 5, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
# Color buttons (bottom row, y: 50-100)
cv2.rectangle(tools, (0, 50), (max_x - ml, max_y),
              (255, 255, 255), -1)  # White background
cv2.rectangle(tools, (0, 50), (max_x - ml, max_y),
              (0, 0, 255), 2)  # Red border
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0), (0, 255, 255)]
for i, x in enumerate([0, 50, 100, 150, 200]):
    cv2.rectangle(tools, (x, 50), (x + 50, max_y), colors[i], -1)

# Initialize 3-channel mask for colored drawings
mask = np.ones((480, 640, 3), dtype='uint8') * 255

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

try:
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        frm = cv2.flip(frm, 1)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        op = hand_landmark.process(rgb)
        if op.multi_hand_landmarks:
            for i in op.multi_hand_landmarks:
                draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
                x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

                # Tool selection (y: 0-50)
                if x < max_x and y < 50 and x > ml:
                    if time_init:
                        ctime = time.time()
                        time_init = False
                    ptime = time.time()
                    cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                    rad -= 1
                    if (ptime - ctime) > tool_select_delay:
                        curr_tool = getTool(x)
                        print(f"Current tool: {curr_tool}")
                        time_init = True
                        rad = 40
                # Color selection (y: 50-100)
                elif x < max_x and y >= 50 and y < max_y and x > ml:
                    if time_init:
                        ctime = time.time()
                        time_init = False
                    ptime = time.time()
                    cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                    rad -= 1
                    if (ptime - ctime) > tool_select_delay:
                        curr_color, color_name = getColor(x)
                        print(f"Current color: {color_name}")
                        time_init = True
                        rad = 40
                else:
                    time_init = True
                    rad = 40

                # Drawing logic
                xi, yi = int(i.landmark[12].x *
                             640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                # Calculate hand centroid for eraser
                centroid_x = int(np.mean([lm.x for lm in i.landmark]) * 640)
                centroid_y = int(np.mean([lm.y for lm in i.landmark]) * 480)

                if curr_tool == "draw" and index_raised(yi, y9):
                    cv2.line(mask, (prevx, prevy), (x, y), curr_color, thick)
                    prevx, prevy = x, y
                elif curr_tool == "line" and index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.line(frm, (xii, yii), (x, y), curr_color, thick)
                elif curr_tool == "rectangle" and index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.rectangle(frm, (xii, yii), (x, y), curr_color, thick)
                elif curr_tool == "circle" and index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    radius = int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                    cv2.circle(frm, (xii, yii), radius, curr_color, thick)
                elif curr_tool == "erase" and is_hand_open(i.landmark):
                    cv2.circle(frm, (centroid_x, centroid_y),
                               50, (0, 0, 0), -1)  # Use centroid
                    cv2.circle(mask, (centroid_x, centroid_y),
                               50, (255, 255, 255), -1)
                else:
                    if var_inits and curr_tool in ["line", "rectangle", "circle"]:
                        if curr_tool == "line":
                            cv2.line(mask, (xii, yii), (x, y),
                                     curr_color, thick)
                        elif curr_tool == "rectangle":
                            cv2.rectangle(mask, (xii, yii),
                                          (x, y), curr_color, thick)
                        elif curr_tool == "circle":
                            radius = int(
                                ((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                            cv2.circle(mask, (xii, yii), radius,
                                       curr_color, thick)
                        var_inits = False
                    prevx, prevy = x, y

        # Apply 3-channel mask
        op = cv2.bitwise_and(frm, mask)
        frm = op

        # Overlay toolbar
        frm[:max_y, ml:max_x] = cv2.addWeighted(
            tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

        # Display current tool and color
        cv2.putText(frm, f"Tool: {curr_tool}", (270 + ml, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frm, f"Color: {color_name}", (270 + ml, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("paint app", frm)

        if cv2.waitKey(1) == 27:
            break

finally:
    cap.release()
    hand_landmark.close()
    cv2.destroyAllWindows()
