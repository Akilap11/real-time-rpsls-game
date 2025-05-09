# 🌟 CGV GAME: Rock, Paper, Scissors Gesture Application 🎮

Interactive gesture-based game developed for **Computer Graphics and Visualization** coursework (2025–2026) at NSBM Green University.

---

## 📝 Project Overview

This project is a real-time **Rock, Paper, Scissors** game using **Python** and **OpenCV**, enabling gesture recognition via webcam. Triggered by the phrase **"Rock, Paper, Scissor, Shoot"**, the application offers a live, interactive experience where players compete against the computer. Designed with educational and entertainment value, it showcases core principles of image processing, computer vision, and UI integration.

---

## ✨ Key Features

- 🎥 **Real-Time Gesture Detection**: Recognizes gestures within a 2-second timeout.
- 🖼️ **Image Processing Pipeline**: Displays steps like grayscale, skin detection, and contour analysis.
- 📊 **Interactive UI**: Real-time score updates, results, and game history shown on video feed.
- 🛠️ **Debugging Support**: Visualizes intermediate image processing steps for learning.
- 🌟 **Future-Ready**: Designed for future expansion like *Rock, Paper, Scissors, Lizard, Spock*.

---

## 🚀 Getting Started

Follow these steps to set up and run the application on your local machine.

### 📋 Prerequisites

- Python 3.8+ installed on your system.
- A working webcam.
- Basic knowledge of Python and Flask.

---

## 🎥 Video Preview

Watch a quick demo of the game in action!  
![Game Demo 1](/media/output.gif) <br><br><br>
![Game Demo 2](/media/output2.gif)

*Click [here](https://www.youtube.com/watch?v=your-video-id) to watch the full video on YouTube (if available).*

*Alternative Link*: If the GIF doesn’t load, you can download the video directly from [media/game-demo.mp4](media/game-demo.mp4).

---

## 🛠️ Setting Up and Running the CGV GAME Application

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/cgv-game.git
   cd cgv-game
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   ```

3. **Set Execution Policy (Windows Only)**
   ```bash
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   ```

4. **Activate the Virtual Environment**

   **Windows:**
   ```bash
   .\venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

5. **Upgrade pip**
   ```bash
   python -m pip install --upgrade pip
   ```

6. **Install Dependencies**
   ```bash
   pip install flask opencv-python numpy
   ```

7. **Run the Application**
   ```bash
   python app.py
   ```

8. **Access the Game**

   Open your browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🎮 How to Play

- Click the **"Play"** button on the web interface.
- Say **"Rock, Paper, Scissor, Shoot"** and show your gesture (rock = fist, paper = open hand, scissors = two fingers) in the green box.
- The result (win, lose, draw) will be displayed along with updated scores and game history.
- Click **"Play"** to start another round.

---

## 🎯 Game Rules

- **Rock** beats **Scissors**
- **Scissors** beats **Paper**
- **Paper** beats **Rock**
- Same gestures = **Draw**

---

## 🖥️ Image Processing Pipeline

Each frame from the webcam is processed through the following steps (visualized on screen):

1. **Grayscale Conversion** – Simplifies the image for analysis.
2. **Gaussian Blur** – Removes noise using a 5x5 kernel.
3. **HSV Conversion** – Converts BGR to HSV for better skin color detection.
4. **Binary Masking** – Applies thresholds to detect hand regions.
5. **Morphological Operations** – Cleans up the binary mask using opening and dilation.
6. **Contour Detection** – Finds hand shape, convex hull, and defects.
7. **Gesture Classification** – Uses finger count, circularity, and aspect ratio to determine the gesture.

---

## 📦 Code Structure

```
cgv-game/
│
├── app.py                # Main Flask server and video stream
├── game_Logic.py           # Game logic, gesture detection, and image processing
├── templates/
│   └── index.html        # Web UI template
├── static/
│   ├── styles.css        # CSS styling
│   └── script.js         # JavaScript interactivity
├── venv/                 # Virtual environment (excluded from Git)
├── README.md             # Project documentation (this file)
└── LICENSE               # Project license
```

---

## 🛠️ Technologies Used

- **Python 3.8+** – Programming language
- **OpenCV** – Image processing and computer vision
- **Flask** – Web framework
- **NumPy** – Array and numerical operations
- **Git/GitHub** – Version control and collaboration

---

## 🐛 Troubleshooting

| Problem                    | Solution                                                                 |
|----------------------------|--------------------------------------------------------------------------|
| Webcam not detected        | Use `cv2.VideoCapture(1)` if `cv2.VideoCapture(0)` doesn't work.         |
| Gesture detection fails    | Ensure good lighting and adjust HSV thresholds in `game_Logic.py`.         |
| Flask server won't start   | Make sure Flask is installed and no other app is using port 5000.        |
| Browser can't connect      | Try [http://localhost:5000](http://localhost:5000).                      |
| Performance lag            | Reduce webcam resolution or simplify image processing pipeline.          |

---

## 📋 Changelog

- **v1.0** (2025-05-09): Initial release with gesture detection and Flask UI./Added 2-second timeout, improved gesture recognition./Game history UI, debug views, minor bug fixes.
                    

---

## 👥 Contributors

| Index Number     | Name                  |
|------------------|-----------------------|
| 22786            | MA Sriyanjith         |
| 22805            | RM Deshapriya         |
| 22791            | WAT Lankathilaka      |
| 22793            | RC Jayalath           |
| 22789            | DK Jayalal            |
| 22883            | KGB Akash             |
---

## 📄 License

This project is licensed under the MIT License.
