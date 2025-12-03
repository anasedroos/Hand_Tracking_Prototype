PROJECT: Hand Tracking Prototype with Virtual Danger Zone
COMPANY: Arvyax (Internship Assignment)

HOW TO RUN:
1. Install Python 3.11 or above
2. Install dependencies:
   pip install -r requirements.txt
3. Run the script:
   python hand_danger_poc.py
4. Press "Q" or "q" to exit the camera window

DESCRIPTION:
The system uses real-time webcam feed and classical computer vision.
It tracks the user's hand using HSV skin color segmentation and contour detection.
A virtual rectangle is drawn in the middle of the screen.
The distance between the hand centroid and the rectangle is measured.
Based on this distance, the system displays:
SAFE (far), WARNING (approaching), DANGER (inside boundary).
In danger state, "DANGER DANGER" is displayed in red.

FILES INCLUDED:
- hand_danger_poc.py
- requirements.txt
