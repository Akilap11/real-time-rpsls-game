from flask import Flask, render_template, Response, jsonify
from game_logic import RockPaperScissors
import logging
import json
import threading
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global game instance
game = None
game_lock = threading.Lock()

def initialize_game():
    """Initialize the game with appropriate error handling."""
    global game
    try:
        with game_lock:
            game = RockPaperScissors()
        logger.debug("Game initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize game: {e}")
        game = None

def gen_frames():
    """Generate video frames from the game's camera feed."""
    global game
    if game is None:
        logger.error("Game object not initialized")
        return

    while True:
        try:
            with game_lock:
                if not game.is_running:
                    logger.debug("Game is no longer running")
                    break
                frame = game.generate_frame()
            
            if frame is None:
                logger.error("No frame generated")
                time.sleep(0.1)
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
            # Control the frame rate
            time.sleep(0.03)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    """Render the game interface."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream video feed to the client."""
    logger.debug("Accessing video feed route")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/game_state')
def game_state():
    """Return the current game state as JSON."""
    global game
    if game is None:
        return jsonify({'error': 'Game not initialized'}), 500
    
    try:
        with game_lock:
            state = game.get_game_state()
        return jsonify(state)
    except Exception as e:
        logger.error(f"Error getting game state: {e}")
        return jsonify({'error': 'Failed to get game state'}), 500

@app.route('/debug_images')
def debug_images():
    """Return debug processing images as base64 encoded strings."""
    global game
    if game is None:
        return jsonify({'error': 'Game not initialized'}), 500
    
    try:
        with game_lock:
            images = game.get_debug_images()
        return jsonify(images)
    except Exception as e:
        logger.error(f"Error getting debug images: {e}")
        return jsonify({'error': 'Failed to get debug images'}), 500
    
@app.route('/game_history')
def game_history():
    """Return the game history as JSON."""
    global game
    if game is None:
        return jsonify({'error': 'Game not initialized'}), 500

    try:
        with game_lock:
            history = list(game.game_history) 
        return jsonify({'history': history})
    except Exception as e:
        logger.error(f"Error getting game history: {e}")
        return jsonify({'error': 'Failed to get game history'}), 500


@app.teardown_appcontext
def cleanup(exception=None):
    """Clean up resources when the application context ends."""
    global game
    if game:
        with game_lock:
            if not game.is_running:
                logger.debug("Cleaning up game resources")
                game.release()
                game = None

if __name__ == '__main__':
    # Initialize game on startup
    initialize_game()
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)