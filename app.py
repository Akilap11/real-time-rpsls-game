from flask import Flask, render_template, Response
from game_logic import RockPaperScissors
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

game = None

def initialize_game():
    global game
    try:
        game = RockPaperScissors()
        logger.debug("Game initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize game: {e}")
        game = None

def gen_frames():
    if game is None:
        logger.error("Game object not initialized")
        return

    while True:
        try:
            frame = game.generate_frame()
            if frame is None:
                logger.error("No frame generated")
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    logger.debug("Accessing video feed route")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.teardown_appcontext
def cleanup(exception=None):
    global game
    if game and not game.is_running:
        logger.debug("Cleaning up game resources")
        game.release()
        game = None

if __name__ == '__main__':
    initialize_game()
    app.run(debug=True, host='0.0.0.0', port=5000)