import cv2
import numpy as np
import math

# basic frame setup
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# virtual rectangle in the middle
RECT_W = 200
RECT_H = 150

rect_x1 = FRAME_WIDTH // 2 - RECT_W // 2
rect_y1 = FRAME_HEIGHT // 2 - RECT_H // 2
rect_x2 = rect_x1 + RECT_W
rect_y2 = rect_y1 + RECT_H

# distance thresholds (in pixels)
FAR_THRESHOLD = 150    # SAFE
NEAR_THRESHOLD = 60    # DANGER

# rough skin color range in HSV
LOWER_SKIN = np.array([0, 30, 60], dtype=np.uint8)
UPPER_SKIN = np.array([20, 150, 255], dtype=np.uint8)


def find_hand_centroid(mask):
    # from the binary mask pick the largest contour as "hand"
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 1000: 
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), largest


def point_to_rect_distance(px, py, x1, y1, x2, y2):
    """
    minimum distance from point (px, py) to the rectangle. returns 0 if the point is inside.
    """
    # inside rectangle
    if x1 <= px <= x2 and y1 <= py <= y2:
        return 0.0

    # horizontal distance
    if px < x1:
        dx = x1 - px
    elif px > x2:
        dx = px - x2
    else:
        dx = 0

    # vertical distance
    if py < y1:
        dy = y1 - py
    elif py > y2:
        dy = py - y2
    else:
        dy = 0

    return math.hypot(dx, dy)


def classify_state(distance):
    # map distance in pixels
    if distance == 0:
        return "DANGER"

    if distance > FAR_THRESHOLD:
        return "SAFE"
    elif distance > NEAR_THRESHOLD:
        return "WARNING"
    else:
        return "DANGER"


def draw_state_overlay(frame, state):
    # simple colored bar at the top showing current state
    if state == "SAFE":
        color = (0, 255, 0)      # green
        text = "SAFE"
    elif state == "WARNING":
        color = (0, 255, 255)    # yellow
        text = "WARNING"
    else:
        color = (0, 0, 255)      # red
        text = "DANGER DANGER"

    cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 40), color, -1)
    cv2.putText(
        frame,
        text,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # mirror the frame so it feels natural
        frame = cv2.flip(frame, 1)

        # HSV + skin mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

        # clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        result = frame.copy()

        # default state
        state = "SAFE"
        distance = None

        # draw the virtual rectangle
        cv2.rectangle(result, (rect_x1, rect_y1),
                      (rect_x2, rect_y2), (255, 0, 0), 2)

        hand_info = find_hand_centroid(mask)

        if hand_info is not None:
            (cx, cy), contour = hand_info

            # hand contour + centroid
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(
                result,
                f"({cx},{cy})",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # distance from hand center to rectangle
            distance = point_to_rect_distance(
                cx, cy, rect_x1, rect_y1, rect_x2, rect_y2
            )

            # line from hand to rectangle center
            rect_cx = (rect_x1 + rect_x2) // 2
            rect_cy = (rect_y1 + rect_y2) // 2
            cv2.line(result, (cx, cy), (rect_cx, rect_cy),
                     (255, 255, 255), 1)

            # classify based on distance
            state = classify_state(distance)

            if distance is not None:
                cv2.putText(
                    result,
                    f"dist: {int(distance)} px",
                    (20, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # overlay SAFE / WARNING / DANGER DANGER on Top
        draw_state_overlay(result, state)

        cv2.imshow("Hand Danger POC - Arvyax", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
