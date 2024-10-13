from FruitNinjaAI import FruitNinjaAI
import pygetwindow as gw
import sys
import keyboard
import time

def main(resolution, training):
    if training:
        ai = FruitNinjaAI('yolo11n.pt', resolution)
        ai.train()

    else:
        ai = FruitNinjaAI(f'runs/detect/{resolution}x{resolution}/weights/best.pt', resolution)

        print(f'Running model {resolution}. Press Enter to start the main loop...')
        input()

        while True:
            if keyboard.is_pressed('esc'):
                print('ESC pressed, stopping the program ...')
                break

            screenshot, offset_x, offset_y = ai.capture(save=True)

            if screenshot is not None:
                detected_objs = ai.predict(screenshot)

                if detected_objs:
                    for _ in range(len(detected_objs)):
                        screenshot, offset_x, offset_y = ai.capture(save=False)
                        detected_objs = ai.predict(screenshot)
                        if not detected_objs or all(obj[0] == 'Bomb' for obj in detected_objs):
                            break
                        # Perform action on the first detected fruit
                        ai.perform_action(detected_objs[0], offset_x, offset_y)
                        time.sleep(0.1)
                else:
                    time.sleep(0.5)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run.py <resolution> <training>")
        sys.exit(1)

    resolution = int(sys.argv[1])
    if resolution not in [256, 320, 640]:
        print("Invalid resolution, defaulting to 320.")
        resolution = 320

    training = '-T' in sys.argv or '-Train' in sys.argv or '-t' in sys.argv or '-train' in sys.argv

    main(resolution, training)