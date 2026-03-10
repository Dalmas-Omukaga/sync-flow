import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO, emit
import json

app = Flask(__name__)
# Allow any origin so your dashboard can also connect easily
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    print("[INFO] Vision Pipeline or Dashboard connected.")

@socketio.on('vision_data')
def handle_vision_data(payload):
    # This matches the 'vision_data' event emitted by your pipeline
    print(f"[LIVE] Focus: {payload.get('focus_score', 0):.1f}% | Label: {payload.get('focus_label')}")
    
    # BROADCAST: Send this data to the Dashboard automatically
    emit('dashboard_update', payload, broadcast=True)

if __name__ == "__main__":
    print("[STARTING] Sync-Flow Socket.io Server on port 5000...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)