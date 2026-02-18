# 🎨 Air Drawing System -- Gesture Controlled Drawing App

A real-time AI-powered virtual drawing application built using
**MediaPipe 0.10+**, **OpenCV**, and **NumPy**.

Control your drawing canvas using only hand gestures --- no mouse, no
touchscreen.

> Built using AI-assisted development in under 25 minutes 🚀

------------------------------------------------------------------------

## ✨ Features

-   ✋ Gesture-Based Controls
-   🎨 Multiple Colors
-   🖌 Adjustable Brush Thickness
-   🧽 Adjustable Eraser Size
-   ↩ Undo Support (Up to 20 steps)
-   💾 Save Drawings as PNG
-   🧼 Clear Canvas
-   📊 Real-time FPS Counter
-   🦴 Hand Skeleton Visualization
-   🎯 Smooth Cursor Tracking

------------------------------------------------------------------------

## 🖐 Gesture Controls

  Gesture               Action
  --------------------- --------------------
  ☝ Index Finger Up     Draw
  ✌ Index + Middle Up   Move Cursor (Idle)
  🖐 All 4 Fingers Up    Erase

------------------------------------------------------------------------

## ⌨ Keyboard Shortcuts

  Key     Function
  ------- ------------------
  `C`     Clear Canvas
  `S`     Save Drawing
  `U`     Undo Last Stroke
  `ESC`   Quit Application

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python 3.9+
-   OpenCV
-   MediaPipe 0.10+ (Task API)
-   NumPy

------------------------------------------------------------------------

## 📦 Installation

Clone the repository:

``` bash
git clone https://github.com/yourusername/air-drawing-system.git
cd air-drawing-system
```

Install dependencies:

``` bash
pip install opencv-python mediapipe numpy
```

------------------------------------------------------------------------

## ▶️ Run the Project

``` bash
python air_drawing.py
```

If the camera does not open, try changing:

``` python
CAMERA_INDEX = 1
```

------------------------------------------------------------------------

## 🧠 How It Works

-   Uses MediaPipe Hand Landmarker (Live Stream Mode)
-   Detects 21 hand landmarks
-   Identifies finger states using tip & PIP joint comparison
-   Classifies gesture into DRAW / IDLE / ERASE
-   Smooths cursor using moving average
-   Maintains undo stack using deque
-   Blends drawing canvas with live camera feed

------------------------------------------------------------------------

## 🏗 Architecture Overview

    Camera → MediaPipe Hand Detection → Gesture Classification
            → Mode Selection → Drawing Engine → UI Rendering

------------------------------------------------------------------------

## 🚀 Why This Project Is Special

Many gesture-based drawing demos look complex and advanced.

This project was built using: - AI-assisted development - Rapid
prototyping - Smart debugging - Model integration

The focus was not typing code ---\
The focus was building a working system efficiently.

This represents: - AI collaboration - Problem-solving mindset - System
thinking - Product thinking

------------------------------------------------------------------------

## 📌 Future Improvements

-   Multi-hand support
-   Shape drawing mode
-   Color selection using gestures
-   Mobile/web version
-   Cloud save support

------------------------------------------------------------------------

## 👨‍💻 Author

**Shashank Kanade**\
BSc Data Science & Business Analytics\
Aspiring Data Scientist \| AI Enthusiast
