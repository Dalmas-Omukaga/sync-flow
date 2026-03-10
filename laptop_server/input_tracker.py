from pynput import keyboard, mouse
import time

key_count = 0
mouse_moves = 0

def on_press(key):
    global key_count
    key_count += 1

def on_move(x, y):
    global mouse_moves
    mouse_moves += 1

keyboard.Listener(on_press=on_press).start()
mouse.Listener(on_move=on_move).start()

while True:

    time.sleep(10)

    typing_speed = key_count / 10
    mouse_entropy = mouse_moves / 10

    print("Typing:", typing_speed)
    print("Mouse:", mouse_entropy)

    key_count = 0
    mouse_moves = 0