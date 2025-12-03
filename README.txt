# Hand Tracking Prototype — Internship Assignment

This project demonstrates real-time hand tracking using classical computer vision techniques (without MediaPipe, OpenPose, or cloud pose APIs).  
A virtual rectangle is drawn on the webcam feed and the system reacts based on the distance between the user's hand and the boundary.

---

## 🚀 Features
✔ Real-time hand detection using HSV + contour tracking  
✔ Calculates hand distance from a virtual rectangular boundary  
✔ 3 safety states displayed live on screen:
- 🟢 SAFE – hand is far from boundary  
- 🟡 WARNING – hand is approaching  
- 🔴 DANGER – hand crosses boundary (shows **"DANGER DANGER"**)  
✔ Runs CPU-only, ≥ 8 FPS

---

## 🧠 Tech Stack
| Component | Technology |
|----------|-------------|
| Language | Python |
| Computer Vision | OpenCV |
| Math / Array Ops | NumPy |

---

## 📌 How to Run the Project
1. Install Python 3.11 or above
2. Install dependencies:
```bash
pip install -r requirements.txt
