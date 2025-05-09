import cv2
import numpy as np
import random
from collections import deque
import logging
import time
import base64
import mediapipe as mp

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

choices = ["rock", "paper", "scissors", "lizard", "spock"]
MAX_HISTORY = 5
GESTURE_CONFIRM_FRAMES = 5  # Frames to confirm gesture

class RockPaperScissors:
    def __init__(self):
        logger.debug("Initializing RockPaperScissors...")
        self.cap = None
        self._initialize_camera()
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.player_score = 0
        self.computer_score = 0
        self.game_history = deque(maxlen=MAX_HISTORY)
        self.player_choice = None
        self.computer_choice = None
        self.result = None
        self.is_running = True
        
        # Game state variables
        self.waiting_for_play = True
        self.waiting_for_gesture = False
        self.showing_result = False
        self.result_timestamp = 0
        
        # Gesture detection smoothing
        self.gesture_buffer = deque(maxlen=GESTURE_CONFIRM_FRAMES)
        
        # Store processing steps for visualization
        self.processing_images = {}
        
        # Frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Centered gesture area (60% of frame width/height)
        size = int(min(self.frame_width, self.frame_height) * 0.6)
        x = (self.frame_width - size) // 2
        y = (self.frame_height - size) // 2
        self.GESTURE_AREA = (x, y, size, size)

    def _initialize_camera(self):
        """Initialize the camera with retry logic."""
        max_retries = 5
        for attempt in range(max_retries):
            for index in range(3):
                logger.debug(f"Attempt {attempt + 1}: Trying camera index {index}...")
                self.cap = cv2.VideoCapture(index)
                if self.cap.isOpened():
                    logger.debug(f"Successfully opened camera at index {index}")
                    return
                self.cap.release()
            time.sleep(1)
        logger.error("Could not open any webcam after multiple retries")
        raise Exception("Could not open webcam")

    def preprocess_frame(self, frame):
        """Process the frame through multiple image processing steps for visualization."""
        # Store original frame
        self.processing_images['original'] = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.processing_images['grayscale'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        self.processing_images['blurred'] = blurred.copy()
        
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        self.processing_images['hsv'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Define range for skin color detection
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create binary mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        self.processing_images['binary_mask'] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        self.processing_images['morph_open'] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        mask = cv2.dilate(mask, kernel, iterations=2)
        self.processing_images['dilated'] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(frame, frame, mask=mask)
        self.processing_images['skin_segmented'] = result.copy()
        
        # Convert BGR to RGB for MediaPipe gesture detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.processing_images['rgb'] = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        return rgb_frame

    def detect_hand_gesture(self, frame, roi, rgb_frame):
        """Detect the hand gesture in the region of interest using MediaPipe."""
        x, y, w, h = roi
        
        # Process frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            cv2.putText(frame, "No hand detected", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None
        
        # Use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw landmarks and connections
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        self.processing_images['landmarks'] = frame.copy()
        
        # Get landmark coordinates
        h, w, _ = frame.shape
        landmarks = []
        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks.append((cx, cy))
        
        # Check if hand is in gesture area
        wrist = landmarks[0]
        if not (x < wrist[0] < x + w and y < wrist[1] < y + h):
            cv2.putText(frame, "Hand outside gesture area", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None
        
        # Finger indices
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]  # Proximal interphalangeal joints
        extended_fingers = []
        
        # Detect extended fingers
        for tip, pip in zip(finger_tips, finger_pips):
            tip_y = landmarks[tip][1]
            pip_y = landmarks[pip][1]
            # For thumb, check x-coordinate due to orientation
            if tip == 4:  # Thumb
                tip_x = landmarks[tip][0]
                pip_x = landmarks[pip][0]
                wrist_x = landmarks[0][0]
                is_right_hand = landmarks[9][0] > landmarks[5][0]  # Middle finger MCP > Index MCP
                if is_right_hand:
                    is_extended = tip_x < pip_x - 20
                else:
                    is_extended = tip_x > pip_x + 20
            else:
                is_extended = tip_y < pip_y - 20
            extended_fingers.append(is_extended)
        
        # Count extended fingers (excluding thumb for some gestures)
        finger_count = sum(extended_fingers[1:])  # Index to Pinky
        thumb_extended = extended_fingers[0]
        
        # Debug info
        debug_text = f"Fingers: {finger_count}, Thumb: {'Up' if thumb_extended else 'Down'}"
        cv2.putText(frame, debug_text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Gesture classification
        detected_gesture = None
        if finger_count == 0 and not thumb_extended:
            detected_gesture = "rock"
        elif finger_count == 4 and thumb_extended:
            detected_gesture = "paper"
        elif finger_count == 2 and extended_fingers[1] and extended_fingers[2]:  # Index and Middle
            detected_gesture = "scissors"
        elif finger_count == 2 and extended_fingers[0] and extended_fingers[4]:  # Thumb and Pinky
            detected_gesture = "lizard"
        elif finger_count == 2 and extended_fingers[1] and extended_fingers[2]:  # Index and Middle, V-shape
            # Check for V-shape (Spock)
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            dist = np.sqrt((index_tip[0] - middle_tip[0])**2 + (index_tip[1] - middle_tip[1])**2)
            if dist > 50:  # Wide V
                detected_gesture = "spock"
        
        if not detected_gesture:
            cv2.putText(frame, "Unknown gesture", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Temporal smoothing
        self.gesture_buffer.append(detected_gesture)
        if len(self.gesture_buffer) == GESTURE_CONFIRM_FRAMES:
            gesture_counts = {}
            for g in self.gesture_buffer:
                if g:
                    gesture_counts[g] = gesture_counts.get(g, 0) + 1
            if gesture_counts:
                confirmed_gesture = max(gesture_counts, key=gesture_counts.get)
                if gesture_counts[confirmed_gesture] >= GESTURE_CONFIRM_FRAMES - 1:
                    return confirmed_gesture
        
        return None

    def determine_winner(self, player_choice, computer_choice):
        """Determine the winner based on Rock-Paper-Scissors-Lizard-Spock rules."""
        if player_choice == computer_choice:
            return "Draw!"
        
        wins = {
            "rock": ["scissors", "lizard"],
            "paper": ["rock", "spock"],
            "scissors": ["paper", "lizard"],
            "lizard": ["paper", "spock"],
            "spock": ["rock", "scissors"]
        }
        
        return "You Win!" if computer_choice in wins[player_choice] else "Computer Wins!"

    def get_debug_images(self):
        """Return base64 encoded debug images."""
        debug_images = {}
        for name, img in self.processing_images.items():
            resized = cv2.resize(img, (150, 150))
            _, buffer = cv2.imencode('.jpg', resized)
            debug_images[name] = base64.b64encode(buffer).decode('utf-8')
        return debug_images
    
    def play_game(self):
        """Start a new game round."""
        logger.debug("Starting new game round")
        self.waiting_for_play = False
        self.waiting_for_gesture = True
        self.showing_result = False
        self.player_choice = None
        self.computer_choice = random.choice(choices)
        logger.debug(f"Computer chose: {self.computer_choice}")
        self.result = None
        self.gesture_buffer.clear()
        
    def get_game_state(self):
        """Return current game state."""
        return {
            'player_score': self.player_score,
            'computer_score': self.computer_score,
            'player_choice': self.player_choice,
            'computer_choice': self.computer_choice if self.showing_result else None,
            'result': self.result,
            'waiting_for_play': self.waiting_for_play,
            'waiting_for_gesture': self.waiting_for_gesture,
            'showing_result': self.showing_result,
            'history': list(self.game_history)
        }

    def generate_frame(self):
        """Generate a processed frame."""
        if not self.is_running:
            logger.warning("Game is not running")
            return None

        if not self.cap or not self.cap.isOpened():
            logger.error("Camera not open, reinitializing")
            self._initialize_camera()

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            return None

        self.processing_images = {}
        frame = cv2.flip(frame, 1)
        
        rgb_frame = self.preprocess_frame(frame)
        
        x, y, w, h = self.GESTURE_AREA
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 6)
        
        self._handle_game_logic(frame, rgb_frame)
        self._display_game_info(frame)
        
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
        
    def _display_game_info(self, frame):
        """Display game information."""
        if self.waiting_for_play:
            cv2.putText(frame, "Press PLAY to start", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif self.waiting_for_gesture:
            cv2.putText(frame, "Show your gesture in the green box", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif self.showing_result:
            cv2.putText(frame, "Result shown! Press PLAY to continue", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Score - You: {self.player_score} Computer: {self.computer_score}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.player_choice and not self.waiting_for_play:
            cv2.putText(frame, f"Detected: {self.player_choice}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                       
        cv2.putText(frame, "Game History:", (frame.shape[1] - 200, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for i, result in enumerate(self.game_history):
            y_pos = 80 + i * 30
            cv2.putText(frame, result, (frame.shape[1] - 200, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.showing_result and self.result:
            for i, text in enumerate([f"Computer: {self.computer_choice}", 
                                   f"You: {self.player_choice}", 
                                   f"Result: {self.result}"]):
                cv2.putText(frame, text, (10, 200+i*30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                          
    def _handle_game_logic(self, frame, rgb_frame):
        """Handle game logic."""
        current_time = time.time()
        logger.debug(f"Game state: waiting_for_gesture={self.waiting_for_gesture}, showing_result={self.showing_result}")
        
        if self.waiting_for_gesture:
            detected_gesture = self.detect_hand_gesture(frame, self.GESTURE_AREA, rgb_frame)
            logger.debug(f"Detected gesture: {detected_gesture}")
            
            if detected_gesture:
                self.player_choice = detected_gesture
                self.result = self.determine_winner(self.player_choice, self.computer_choice)
                
                if self.result == "You Win!":
                    self.player_score += 1
                elif self.result == "Computer Wins!":
                    self.computer_score += 1
                
                history_entry = f"You: {self.player_choice}, Comp: {self.computer_choice}, {self.result}"
                self.game_history.append(history_entry)
                
                self.waiting_for_gesture = False
                self.showing_result = True
                self.result_timestamp = current_time
        
        if self.showing_result and current_time - self.result_timestamp > 3:
            self.showing_result = False
            self.waiting_for_play = True
    
    def release(self):
        """Release camera and MediaPipe resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.debug("Camera resources released")
        self.hands.close()
        self.is_running = False