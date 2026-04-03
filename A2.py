import cv2
import numpy as np
import mediapipe as mp
import math

# ================= CONFIG =================
HEADER_HEIGHT = 80
BRUSH = 8
ERASER_SIZE = 70

# Unified Tools List (Name, Color, Type)
TOOLS = [
    ("PURP", (255, 0, 255), "COLOR"),
    ("BLUE", (255, 0, 0), "COLOR"),
    ("GRN", (0, 255, 0), "COLOR"),
    ("YLW", (0, 255, 255), "COLOR"),
    ("ERAS", (0, 0, 0), "COLOR"),
    ("FREE", (150, 150, 150), "SHAPE"),
    ("LINE", (150, 150, 150), "SHAPE"),
    ("RECT", (150, 150, 150), "SHAPE"),
    ("CIRC", (150, 150, 150), "SHAPE"),
    ("TRI", (150, 150, 150), "SHAPE")
]

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# ================= STATE VARIABLES =================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = None
current_color = TOOLS[1][1] # Default Blue
current_shape = "FREE"

# Drawing state tracking
prev_point = None
shape_start = None
current_point = None
is_drawing_shape = False

# Undo / Redo Memory Stacks
undo_stack = []
redo_stack = []
MAX_UNDO_STEPS = 20
was_modifying_canvas = False # Tracks continuous stroke states

screenshot_count = 1

# ================= GESTURES & HELPERS =================
def is_index_up(lm): return lm[8].y < lm[6].y
def is_middle_up(lm): return lm[12].y < lm[10].y
def is_ring_up(lm): return lm[16].y < lm[14].y
def is_pinky_up(lm): return lm[20].y < lm[18].y

def is_closed_fist(lm):
    palm = lm[0]
    def dist(a, b): return math.hypot(a.x - b.x, a.y - b.y)
    threshold = 0.12
    return (dist(lm[8], palm) < threshold and dist(lm[12], palm) < threshold and
            dist(lm[16], palm) < threshold and dist(lm[20], palm) < threshold)

def draw_shape_overlay(img, start, end, shape_type, color, thickness):
    """Handles the geometric rendering for shapes"""
    if start is None or end is None: return
    x1, y1 = start
    x2, y2 = end

    if shape_type == "LINE":
        cv2.line(img, start, end, color, thickness)
    elif shape_type == "RECT":
        cv2.rectangle(img, start, end, color, thickness)
    elif shape_type == "CIRC":
        radius = int(math.hypot(x2 - x1, y2 - y1))
        cv2.circle(img, start, radius, color, thickness)
    elif shape_type == "TRI":
        pts = np.array([[(x1 + x2) // 2, y1], [x1, y2], [x2, y2]], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, thickness)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret: continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    mode = "IDLE"
    hand_count = 0
    index_only = False
    is_modifying_canvas = False # Reset flag for current frame
    x, y = 0, 0

    # ================= HAND PROCESS =================
    if result.multi_hand_landmarks:
        hand_count = len(result.multi_hand_landmarks)
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        lm = hand.landmark
        x, y = int(lm[8].x * w), int(lm[8].y * h)

        index = is_index_up(lm)
        middle = is_middle_up(lm)
        ring = is_ring_up(lm)
        pinky = is_pinky_up(lm)

        box_w = w // len(TOOLS)

        # 1. ERASE MODE
        if is_closed_fist(lm):
            mode = "ERASER (FIST)"
            is_modifying_canvas = True
            
            # Record state before erasing starts
            if not was_modifying_canvas:
                undo_stack.append(canvas.copy())
                redo_stack.clear()
                if len(undo_stack) > MAX_UNDO_STEPS: undo_stack.pop(0)
                
            cv2.circle(canvas, (x, y), ERASER_SIZE, (0, 0, 0), -1)
            is_drawing_shape = False
            shape_start = None

        # 2. SELECTION MODE
        elif index and middle and not ring and not pinky:
            mode = "SELECTING"
            if y < HEADER_HEIGHT:
                idx = min(x // box_w, len(TOOLS) - 1)
                name, col, t_type = TOOLS[idx]
                if t_type == "COLOR":
                    current_color = col
                elif t_type == "SHAPE":
                    current_shape = name
            is_drawing_shape = False
            shape_start = None

        # 3. DRAWING MODE
        elif index and not middle and not ring and not pinky:
            mode = "DRAWING"
            index_only = True

    # ================= DRAWING STATE MACHINE =================
    if index_only:
        current_point = (x, y)
        if current_shape != "FREE":
            if shape_start is None:
                shape_start = (x, y)
            is_drawing_shape = True
        else:
            # Freestyle drawing
            is_modifying_canvas = True
            if not was_modifying_canvas:
                undo_stack.append(canvas.copy())
                redo_stack.clear()
                if len(undo_stack) > MAX_UNDO_STEPS: undo_stack.pop(0)

            if prev_point is None: prev_point = (x, y)
            cv2.line(canvas, prev_point, (x, y), current_color, BRUSH if current_color != (0, 0, 0) else ERASER_SIZE)
            prev_point = (x, y)
    else:
        # Commit shape to canvas when finger drops
        if is_drawing_shape and shape_start is not None and current_point is not None:
            undo_stack.append(canvas.copy())
            redo_stack.clear()
            if len(undo_stack) > MAX_UNDO_STEPS: undo_stack.pop(0)
            
            draw_shape_overlay(canvas, shape_start, current_point, current_shape, 
                               current_color, BRUSH if current_color != (0, 0, 0) else ERASER_SIZE)
        
        # Reset states
        is_drawing_shape = False
        shape_start = None
        current_point = None
        prev_point = None

    # Update continuous tracking flag
    was_modifying_canvas = is_modifying_canvas

    # ================= MERGE CANVAS & FRAME =================
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    # ================= DRAW SHAPE PREVIEW =================
    if is_drawing_shape and shape_start is not None and current_point is not None:
        draw_shape_overlay(frame, shape_start, current_point, current_shape, 
                           current_color, BRUSH if current_color != (0, 0, 0) else ERASER_SIZE)

    # ================= RENDER HEADER UI =================
    box_w = w // len(TOOLS)
    for i, (name, color, t_type) in enumerate(TOOLS):
        cv2.rectangle(frame, (i * box_w, 0), ((i + 1) * box_w, HEADER_HEIGHT), color, -1)
        
        is_active_color = (t_type == "COLOR" and color == current_color and current_shape == "FREE")
        is_active_shape = (t_type == "SHAPE" and name == current_shape)
        
        if is_active_color or is_active_shape:
            cv2.rectangle(frame, (i * box_w, 0), ((i + 1) * box_w, HEADER_HEIGHT), (0, 0, 255), 4)
        else:
            cv2.rectangle(frame, (i * box_w, 0), ((i + 1) * box_w, HEADER_HEIGHT), (255, 255, 255), 2)

        text_color = (255, 255, 255) if (color == (0, 0, 0) or t_type == "COLOR") else (0, 0, 0)
        cv2.putText(frame, name, (i * box_w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # ================= RENDER STATUS UI =================
    status_top = HEADER_HEIGHT + 20
    cv2.rectangle(frame, (20, status_top), (350, status_top + 180), (40, 40, 40), -1)
    cv2.rectangle(frame, (20, status_top), (350, status_top + 180), (200, 200, 200), 2)

    color_name = next(name for name, col, t in TOOLS if t == "COLOR" and col == current_color)

    cv2.putText(frame, "AIR CANVAS", (35, status_top + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"MODE: {mode}", (35, status_top + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"COLOR: {color_name}", (35, status_top + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"SHAPE: {current_shape}", (35, status_top + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"HANDS: {hand_count}", (35, status_top + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ================= RENDER INSTRUCTIONS UI =================
    instr_top = status_top + 195
    instr_bottom = instr_top + 215

    cv2.rectangle(frame, (20, instr_top), (350, instr_bottom), (30, 30, 30), -1)
    cv2.rectangle(frame, (20, instr_top), (350, instr_bottom), (200, 200, 200), 2)

    cv2.putText(frame, "INSTRUCTIONS", (35, instr_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Index Finger  -> Drag / Draw", (35, instr_top + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Open Hand     -> Commit Shape", (35, instr_top + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
    cv2.putText(frame, "Fist          -> Erase / Cancel", (35, instr_top + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, "U -> Undo     | R -> Redo", (35, instr_top + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "C -> Clear    | S -> Screenshot", (35, instr_top + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "Q -> Quit Application", (35, instr_top + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("Air Canvas - Verified Edition", frame)

    # ================= KEYBOARD INPUT CONTROLS =================
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'): break
    
    elif key == ord('c'):
        # Save state before clearing so user can undo an accidental clear
        undo_stack.append(canvas.copy())
        redo_stack.clear()
        if len(undo_stack) > MAX_UNDO_STEPS: undo_stack.pop(0)
        canvas[:] = 0
        
    elif key == ord('s'):
        filename = f"air_canvas_screenshot_{screenshot_count}.png"
        cv2.imwrite(filename, frame)
        print("Screenshot saved:", filename)
        screenshot_count += 1
        
    elif key == ord('u'):
        if len(undo_stack) > 0:
            redo_stack.append(canvas.copy())
            canvas = undo_stack.pop()
            
    elif key == ord('r'):
        if len(redo_stack) > 0:
            undo_stack.append(canvas.copy())
            canvas = redo_stack.pop()

cap.release()
cv2.destroyAllWindows()qq