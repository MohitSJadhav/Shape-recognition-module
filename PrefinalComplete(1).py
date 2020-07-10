#!/usr/bin/env python
# coding: utf-8

# In[25]:


from flask import Flask, render_template, request
import base64
import re
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
from PIL import Image
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import statistics
from statistics import mode
#from shapeprefinal import ShapeRecognition
list1=list()
    

class ShapeRecognition(object):
    def __init__(self, img):
        self.img = img
        self.contours = None
        self.binary_img = None
        self.debug_img = img.copy()

    def get_binary_image(self, lower=[0, 0, 0], upper=[15, 15, 15]):
        self.binary_img = cv2.inRange(
            self.img, np.array(lower), np.array(upper))
        kernel = np.ones((5, 5), np.uint8)
        self.binary_img = cv2.dilate(self.binary_img, kernel, iterations=1)
        return self.binary_img

    def get_contours(self):
        _,self.contours, h = cv2.findContours(
            self.binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        return self.contours

    def draw_debug(self, contour, cx, cy, shape_class):
        cv2.drawContours(self.debug_img, [contour], 0, (0, 0, 255), 3)
        cv2.putText(
            self.debug_img,
            shape_class,
            (cx - 35, cy + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def unit_vector(self, v):
        return v / np.linalg.norm(v)

    def get_corner_angle(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p1[0] - p3[0], p1[1] - p3[1]])
        v1_unit = self.unit_vector(v1)
        v2_unit = self.unit_vector(v2)
        radians = np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1, 1))
        return math.degrees(radians)

    def find_shapes(self, epsilon_factor=0.02):
        self.get_binary_image()
        self.get_contours()
        for n, cnt in enumerate(self.contours):
            approx = cv2.approxPolyDP(
                cnt, epsilon_factor * cv2.arcLength(cnt, True), True
            )
            M = cv2.moments(approx)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            p1 = approx[0][0]
            p2 = approx[1][0]
            p3 = approx[-1][0]
            if len(approx) == 3:  # its a triangle!
                self.draw_debug(cnt, cx, cy, "Triangle")
                list1.append("TRIANGLE")
            if len(approx) == 4:
                degrees = self.get_corner_angle(p1, p2, p3)
                if degrees == 90:
                    self.draw_debug(cnt, cx, cy, "Rectangle")
                    list1.append("RECTANGLE")
                else:
                    self.draw_debug(cnt, cx, cy, "Quadrilateral")
                    list1.append("QUADRILATERAL")
                    print(list1)
            if len(approx) >=7:
                self.draw_debug(cnt, cx, cy, "Circle")
                list1.append("CIRCLE")
                
app = Flask(__name__, template_folder='templates')
    
@app.route('/', methods=['GET'])
def canvas():
    return render_template('index.html')
    
@app.route('/classify', methods=['POST'])
def classify():
    data = request.form.get('imageBase64')
    data = data.replace('data:image/png;base64,', '')
    data = data.replace(' ', '+')
    
    with open('upload.png', 'wb') as fh:
        fh.write(base64.b64decode(data))
    image = Image.open(r"upload.png")
    new_img = Image.new("RGB", (image.size[0],image.size[1]), (255, 255, 255))
    cmp_img = Image.composite(image, new_img, image).quantize(colors=256, method=2)

    cmp_img.save("Destination_path.png")
    
    #--------Img classification with opencv done here!!------------
    img = cv2.imread("Destination_path.png")
    shape_recognition = ShapeRecognition(img)
    shape_recognition.find_shapes()
    combined = np.vstack(
        (
            shape_recognition.debug_img,
            cv2.cvtColor(shape_recognition.binary_img, cv2.COLOR_GRAY2BGR),
        )
    )
    cv2.imwrite("flaskcombined.png",combined)
    #plt.imshow(combined, cmap='gray')
    #plt.show()
    print(list1)
    return render_template('index.html')
@app.route('/about')
def about():
    max_data=mode(list1)
    list1.clear()
    print(max_data)
    return render_template('about.html',max_data=max_data)

if __name__ == "__main__":
    run_simple('localhost',9000,app)
    
        


# In[ ]:





# In[ ]:





# In[ ]:




