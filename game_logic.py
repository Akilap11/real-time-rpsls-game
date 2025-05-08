import cv2
import numpy as np
import random
from collections import deque
import logging
import time
import base64
from collections import deque, Counter

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
        
        # Store processing steps for visualization
        self.processing_images = {}

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
        """Process the frame through multiple image processing steps for hand detection."""
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
        
        return result, mask

    def detect_hand_gesture(self, frame, roi, mask):
        """Detect the hand gesture (rock, paper, scissors) in the region of interest."""
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        
        # Store ROI for debugging
        self.processing_images['roi'] = roi_frame.copy()
        
        # Find contours in ROI
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (assuming it's the hand)
            max_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(max_contour)
            
            # Store contour visualization
            contour_vis = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.drawContours(contour_vis, [max_contour], -1, (0, 255, 0), 2)
            self.processing_images['contour'] = contour_vis
            
            if contour_area > 2000:  # Filter small contours
                # Get bounding rectangle
                rx, ry, rw, rh = cv2.boundingRect(max_contour)
                cv2.rectangle(frame, (x+rx, y+ry), (x+rx+rw, y+ry+rh), (0, 255, 0), 2)
                
                # Find convex hull and convexity defects for finger counting
                hull = cv2.convexHull(max_contour, returnPoints=False)
                defects = cv2.convexityDefects(max_contour, hull)
                
                # Convex hull visualization
                hull_vis = np.zeros((h, w, 3), dtype=np.uint8)
                hull_points = cv2.convexHull(max_contour, returnPoints=True)
                cv2.drawContours(hull_vis, [hull_points], -1, (0, 255, 0), 2)
                self.processing_images['convex_hull'] = hull_vis
                
                finger_count = 0
                if defects is not None:
                    # Defects visualization
                    defects_vis = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.drawContours(defects_vis, [max_contour], -1, (255, 255, 255), 1)
                    
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        far = tuple(max_contour[f][0])
                        end = tuple(max_contour[e][0])
                        
                        # Calculate angle between fingers
                        a = np.linalg.norm(np.array(end) - np.array(start))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(end) - np.array(far))
                        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c)) * 57.2958
                        
                        # Draw defect points
                        cv2.circle(defects_vis, far, 5, [0, 0, 255], -1)
                        
                        # Count fingers based on angle
                        if angle <= 90 and d > 2000:
                            finger_count += 1
                            cv2.circle(defects_vis, far, 5, [0, 255, 0], -1)
                    
                    self.processing_images['defects'] = defects_vis
                    finger_count += 1  # Add 1 for the finger tip
                    
                cv2.putText(frame, f"Fingers: {finger_count}", (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Determine gesture based on finger count and contour area
                if finger_count == 1 or contour_area < 5000:
                    return "rock"
                elif finger_count == 2:
                    return "scissors"
                elif finger_count >= 3 and contour_area > 10000:
                    return "paper"
        
        return None

    def determine_winner(self, player_choice, computer_choice):
        """Determine the winner based on classic rock-paper-scissors rules."""
        wins = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
        if player_choice == computer_choice:
            return "Draw!"
        return "You Win!" if wins[player_choice] == computer_choice else "Computer Wins!"

    def get_debug_images(self):
        """Return base64 encoded debug images for visualization."""
        debug_images = {}
        for name, img in self.processing_images.items():
            # Resize image to a smaller size for faster transmission
            resized = cv2.resize(img, (150, 150))
            _, buffer = cv2.imencode('.jpg', resized)
            debug_images[name] = base64.b64encode(buffer).decode('utf-8')
        return debug_images
    
    def play_game(self):
        """Start a new game round with computer's choice."""
        logger.debug("Starting new game round")
        self.waiting_for_play = False
        self.waiting_for_gesture = True
        self.showing_result = False
        self.player_choice = None
        # Make computer's choice immediately when play is clicked
        self.computer_choice = random.choice(choices)
        logger.debug(f"Computer chose: {self.computer_choice}")
        self.result = None
        
    def get_game_state(self):
        """Return current game state as dictionary."""
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
        """Generate a processed frame for display."""
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

        # Reset processing images dictionary
        self.processing_images = {}

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process the frame
        processed_frame, mask = self.preprocess_frame(frame)
        
        # Draw gesture area rectangle
        x, y, w, h = GESTURE_AREA
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Handle gameplay logic
        self._handle_game_logic(frame, mask)
        
        # Display game information on frame
        self._display_game_info(frame)
        
        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        return frame_bytes
        
    def _display_game_info(self, frame):
        """Display game information on the frame."""
        # Display game state
        if self.waiting_for_play:
            cv2.putText(frame, "Press PLAY to start", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif self.waiting_for_gesture:
            cv2.putText(frame, "Show your gesture in the green box", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif self.showing_result:
            cv2.putText(frame, "Result shown! Press PLAY to continue", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display scores
        cv2.putText(frame, f"Score - You: {self.player_score} Computer: {self.computer_score}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display detected gesture if available
        if self.player_choice and not self.waiting_for_play:
            cv2.putText(frame, f"Detected: {self.player_choice}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                       
        # Display game history
        cv2.putText(frame, "Game History:", (frame.shape[1] - 200, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for i, result in enumerate(self.game_history):
            y_pos = 80 + i * 30
            cv2.putText(frame, result, (frame.shape[1] - 200, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display current game result if available
        if self.showing_result and self.result:
            for i, text in enumerate([f"Computer: {self.computer_choice}", 
                                   f"You: {self.player_choice}", 
                                   f"Result: {self.result}"]):
                cv2.putText(frame, text, (10, 170+i*30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                          
    def _handle_game_logic(self, frame, mask):
        """Handle game logic based on recent gesture consistency."""
        current_time = time.time()
        

        if not hasattr(self, 'recent_gestures'):
            self.recent_gestures = deque(maxlen=5)

        if self.waiting_for_gesture:
            detected_gesture = self.detect_hand_gesture(frame, GESTURE_AREA, mask)

            # Append gesture (or None) to history
            self.recent_gestures.append(detected_gesture)

            logger.debug(f"Recent gestures: {list(self.recent_gestures)}")


            # Count consistent gestures in recent frames
            if len(self.recent_gestures) == self.recent_gestures.maxlen:
                counts = Counter(self.recent_gestures)
                most_common, freq = counts.most_common(1)[0]

                if most_common and freq >= 4:
                    self.player_choice = most_common
                    self.computer_choice = random.choice(choices)
                    self.result = self.determine_winner(self.player_choice, self.computer_choice)

                    # Update scores
                    if self.result == "You Win!":
                        self.player_score += 1
                    elif self.result == "Computer Wins!":
                        self.computer_score += 1

                    # Log history
                    self.game_history.append(
                        f"You: {self.player_choice}, Comp: {self.computer_choice}, {self.result}"
                    )

                    # Reset state
                    self.waiting_for_gesture = False
                    self.showing_result = True
                    self.result_timestamp = current_time
                    self.recent_gestures.clear()

        # After showing result, reset to waiting
        if self.showing_result and (current_time - self.result_timestamp) > 3:
            self.showing_result = False
            self.waiting_for_play = True
    
    def release(self):
        """Release camera resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.debug("Camera resources released")
        self.is_running = False

    