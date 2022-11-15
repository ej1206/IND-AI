import cv2
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default="/home/ej/Downloads/Lenna.png")
parser.add_argument('--outputPath', default="/home/ej/Downloads/Lenna_Result.png")
params = parser.parse_args()


img = cv2.imread(params.path)
w, h = img.shape[1], img.shape[0]

mouse_pressed = False
s_x = s_y = e_x = e_y = -1
img_to_show = np.copy(img)

def rand_pt(mult=1.):
    return(random.randrange(int(w * mult)),
           random.randrange(int(h * mult)))

def mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        print("mouse pressed")
        mouse_pressed = True
        s_x, s_y = x, y
        image_to_show = np.copy(img)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            print("mouse move")

    elif event == cv2.EVENT_LBUTTONUP:
        e_x, e_y = x, y
        #if flags:
        #    cv2.line(img, (s_x, s_y), (e_x, e_y), (0, 255, 0), 4, cv2.LINE_AA)
        #    cv2.imshow("img", img)
        print("mouse up")
        mouse_pressed = False




cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("img", mouse_callback, img)


finish = False
while not finish:
    cv2.imshow("img", img_to_show)

    key = cv2.waitKey(0)

    if key == ord('r'):
        cv2.rectangle(img_to_show, (s_x, s_y), (e_x, e_y), (0, 255, 0), 4)

    elif key == ord('l'):
        cv2.line(img_to_show, (s_x, s_y), (e_x, e_y), (0, 255, 0), 4, cv2.LINE_AA)

    elif key == ord('a'):
        cv2.arrowedLine(img_to_show, (s_x, s_y), (e_x, e_y), (0, 255, 0), 4, cv2.LINE_AA)

    elif key == ord('w'):
        cv2.imwrite(params.outputPath, img_to_show)

    elif key == ord('c'):
        img_to_show = np.copy(img)

    elif key == 27:
        finish = True

cv2.destroyAllWindows()