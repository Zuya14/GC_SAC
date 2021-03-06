import os
import sys
import glob
import cv2

if __name__ == '__main__':
    
    assert len(sys.argv) == 2

    arg = sys.argv[1]

    if os.path.isdir(arg):
        arg = glob.glob(arg + '*.mp4')[0]

    # print(arg)

    cap = cv2.VideoCapture(arg)

    window_name = sys.argv[1]
    delay = int(1000/cap.get(cv2.CAP_PROP_FPS))

    if not cap.isOpened():
        sys.exit()

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.destroyWindow(window_name)