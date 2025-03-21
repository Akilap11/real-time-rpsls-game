import cv2
import numpy as np
import random
from collections import deque
import logging
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

choices = ["rock", "paper", "scissors"]
GESTURE_AREA = (100, 100, 400, 400)
MAX_HISTORY = 5

class RockPaperScissors:
    def __init__(self):
        logger.debug("Initializing RockPaperScissors...")
        self.cap = None
        self._initialize_camera()
        
        self.player_score = 0
        self.computer_score = 0
        self.countdown = 3
        self.last_countdown = 0
        self.game_history = deque(maxlen=MAX_HISTORY)
        self.player_choice = None
        self.computer_choice = None
        self.result = None
        self.is_running = True  

    def _initialize_camera(self):
        """Initialize the camera with retry logic."""
        for index in range(3):  # Try camera indices 0, 1, 2
            logger.debug(f"Trying camera index {index}...")
            self.cap = cv2.VideoCapture(index)
            if self.cap.isOpened():
                logger.debug(f"Successfully opened camera at index {index}")
                return
            self.cap.release()
        logger.error("Could not open any webcam after trying indices 0-2")
        raise Exception("Could not open webcam")

    def preprocess_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
        
        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result, mask

    def detect_hand_gesture(self, frame, roi, mask):
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(max_contour)
            
            if contour_area > 2000:
                rx, ry, rw, rh = cv2.boundingRect(max_contour)
                cv2.rectangle(frame, (x+rx, y+ry), (x+rx+rw, y+ry+rh), (0, 255, 0), 2)
                
                hull = cv2.convexHull(max_contour, returnPoints=False)
                defects = cv2.convexityDefects(max_contour, hull)
                
                finger_count = 0
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        far = tuple(max_contour[f][0])
                        end = tuple(max_contour[e][0])
                        
                        a = np.linalg.norm(np.array(end) - np.array(start))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(end) - np.array(far))
                        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c)) * 57.2958
                        
                        if angle <= 90 and d > 2000:
                            finger_count += 1
                    finger_count += 1
                    
                cv2.putText(frame, f"Fingers: {finger_count}", (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if finger_count == 1 or contour_area < 5000:
                    return "rock"
                elif finger_count == 2:
                    return "scissors"
                elif finger_count >= 3 and contour_area > 10000:
                    return "paper"
        return None

    def determine_winner(self, player_choice, computer_choice):
        wins = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
        if player_choice == computer_choice:
            return "Draw!"
        return "You Win!" if wins[player_choice] == computer_choice else "Computer Wins!"

    def generate_frame(self):
        if not self.is_running:
            logger.warning("Game is not running, cannot generate frame")
            return None

        if not self.cap or not self.cap.isOpened():
            logger.error("Camera is not open, attempting to reinitialize")
            self._initialize_camera()

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture frame from webcam")
            return None

        frame = cv2.flip(frame, 1)
        processed_frame, mask = self.preprocess_frame(frame)
        x, y, w, h = GESTURE_AREA
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        self.player_choice = self.detect_hand_gesture(frame, GESTURE_AREA, mask)
        
        cv2.putText(frame, f"Time: {self.countdown}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Score - You: {self.player_score} Computer: {self.computer_score}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, "Game History:", (frame.shape[1] - 200, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for i, result in enumerate(self.game_history):
            y_pos = 80 + i * 30
            cv2.putText(frame, result, (frame.shape[1] - 200, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        current_time = time.time()
        if self.last_countdown == 0:
            self.last_countdown = current_time

        if self.countdown <= 0:
            self.computer_choice = random.choice(choices)
            if self.player_choice:
                self.result = self.determine_winner(self.player_choice, self.computer_choice)
                if self.result == "You Win!":
                    self.player_score += 1
                elif self.result == "Computer Wins!":
                    self.computer_score += 1
                
                history_entry = f"You: {self.player_choice}, Comp: {self.computer_choice}, {self.result}"
                self.game_history.append(history_entry)
                
                for i, text in enumerate([f"Computer: {self.computer_choice}", 
                                       f"You: {self.player_choice}", 
                                       f"Result: {self.result}"]):
                    cv2.putText(frame, text, (10, 170+i*30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No gesture detected!", (10, 170), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.countdown = 3
            time.sleep(1.5)
        elif current_time - self.last_countdown >= 1:
            self.countdown -= 1
            self.last_countdown = current_time

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        logger.debug("Frame generated successfully")
        return frame

    def release(self):
        logger.debug("Releasing webcam...")
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None