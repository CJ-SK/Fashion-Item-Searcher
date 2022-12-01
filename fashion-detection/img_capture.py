import ImageGrab
import keyboard
import mouse
import numpy as np
import cv2 as cv
import requests
from threading import Timer
import webbrowser
import os
import sys

endpoint = 'http://13.57.247.28:5000/'

def set_roi():
        global ROI_SET, x1, y1, x2, y2
            ROI_SET = False
                print("drag mouse to select!!")
                    while(mouse.is_pressed() == False):
                                x1, y1 = mouse.get_position()
                                        while(mouse.is_pressed() == True):
                                                        x2, y2 = mouse.get_position()
                                                                    while(mouse.is_pressed() == False):
                                                                                        print('coord', x1, y1, x2, y2)
                                                                                                        ROI_SET = True
                                                                                                                        return

                                                                                                                    def open_browser():
                                                                                                                              webbrowser.open_new(endpoint)

                                                                                                                              #keyboard.add_hotkey('enter', lambda: set_roi())
                                                                                                                              set_roi()
                                                                                                                              ROI_SET = False
                                                                                                                              x1, y1, x2, y2 = 0, 0, 0, 0

                                                                                                                              while True:
                                                                                                                                      if ROI_SET == True:
                                                                                                                                                  image = ImageGrab.grab(bbox=(x1, y1, x2, y2))
                                                                                                                                                          result_img_path = './file.jpg'
                                                                                                                                                                  files = cv.imwrite(result_img_path, np.array(image))
                                                                                                                                                                          upload = {'file': files}
                                                                                                                                                                                  res = requests.post(endpoint+'/detect/', files = upload)
                                                                                                                                                                                          Timer(1, open_browser).start()
                                                                                                                                                                                          i
