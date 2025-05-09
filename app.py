from flask import Flask, render_template, Response, jsonify
from game_logic import RockPaperScissors
import logging
import threading
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

game = None
game_lock = threading.Lock()

def initialize_game():
    global game
    try:
        with game_lock:
            game = RockPaperScissors()
        logger.debug("Game initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize game: {e}")
        game = None

def gen_frames():
    global game
    if game is None:
        logger.error("Game object not initialized")
        return

    while True:
        try:
            with game_lock:
                if not game.is_running:
                    break
                frame = game.generate_frame()

            if frame is None:
                time.sleep(0.1)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # ~30 FPS

        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/game_state')
def game_state():
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

@app.route('/play_game', methods=['POST'])
def play_game():
    global game
    if game is None:
        return jsonify({'error': 'Game not initialized'}), 500

    try:
        with game_lock:
            game.play_game()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error starting game: {e}")
        return jsonify({'error': 'Failed to start game'}), 500

@app.teardown_appcontext
def cleanup(exception=None):
    global game
    if game:
        with game_lock:
            if not game.is_running:
                game.release()
                game = None

if __name__ == '__main__':
    initialize_game()
    app.run(debug=True, host='0.0.0.0', port=5000)