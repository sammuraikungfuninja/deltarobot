# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:40:47 2023

@author: daanb
"""

import cv2 
import time 
import numpy as np 
import math
import snap7  
import struct 

plc = snap7.client.Client()  
plc.connect('192.168.0.81', 0, 1)  
def write_int_db(db_number, start_address, value):  
    plc.db_write(db_number, start_address, bytearray(struct.pack('>H', value)))
    return None

height = 1080
width = 1920
cv2.namedWindow("beeld1",cv2.WINDOW_AUTOSIZE)

cv2.moveWindow("beeld1",0,0) 
webcam = cv2.VideoCapture(0) 

webcam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,height) 

while True:
    retval, img = webcam.read()
    if(retval == True):
        cv2.imshow("beeld1",img) 
        cv2.imwrite("afbeelding.jpg",img)
        if(cv2.waitKey(1) == 27): 
            break 
webcam.release() 
cv2.destroyAllWindows()

img = cv2.imread(r'C:\Users\daanb\Documents\python\afbeelding.jpg')

x, y, width, height = 850, 500, 250, 250
roi = img[y:y+height, x:x+width]
img_zoomed_in = cv2.resize(roi, (width * 2, height * 2))

img_gray = cv2.cvtColor(img_zoomed_in,cv2.COLOR_BGR2GRAY)

#img_contrast = np.clip(img_gray * 2, 0, 255).astype(np.uint8)

img_blur = cv2.medianBlur(img_gray, 11)

_, img_binary = cv2.threshold(img_blur, 23, 255, cv2.THRESH_BINARY)

low_threshold = 40
high_threshold = 100
img_edges = cv2.Canny(img_binary, low_threshold, high_threshold)

_, thresh = cv2.threshold(img_edges, 128, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = img_zoomed_in.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
img_show = img_zoomed_in.copy()

center_x = 0
center_y = 0
angle = 0
for contour in contours:
    area = cv2.contourArea(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
#    print(area)
    num_vertices = len(approx)
    if area > 600:
        shape_label = "onbekend"
        
        if num_vertices == 3:
            shape_label = "driehoek"
            vertices = approx.reshape(-1, 2)
            center_x = int(np.mean(vertices[:, 0]))
            center_y = int(np.mean(vertices[:, 1]))
            cv2.circle(img_show, (center_x, center_y), 5, (0, 0, 255), -1)
            
            img_zoomed_in = img_zoomed_in[:, :, :3]
            center = (center_x, center_y)  
            radius = 25  
            yy, xx = np.ogrid[:img_zoomed_in.shape[0], :img_zoomed_in.shape[1]]
            mask = (yy - center[1])**2 + (xx - center[0])**2 <= radius**2
            roi = img_zoomed_in[mask]
            average_color = np.mean(roi, axis=0)
            average_color = tuple(map(int, average_color))
            B, G, R = average_color
#            print(average_color)
            
            """authors: Ilmar & Daan"""   
            vertices = approx.reshape(-1, 2)
            center_x = int(np.mean(vertices[:, 0]))
            center_y = int(np.mean(vertices[:, 1]))
            cv2.circle(img_show, (center_x, center_y), 5, (0, 0, 255), -1)
        
            distances = [np.linalg.norm(vertices[i] - vertices[j]) for i in range(3) for j in range(i+1, 3)]
        
            if distances:
                longest_side_index = np.argmax(distances)
        
                other_vertices_indices = [i for i in range(3) if i != longest_side_index]
        
                center_x_longest_side = int(np.mean([vertices[longest_side_index][0], vertices[other_vertices_indices[1]][0]]))
                center_y_longest_side = int(np.mean([vertices[longest_side_index][1], vertices[other_vertices_indices[1]][1]]))
        
                print(f"{center_x_longest_side},{center_y_longest_side}") #lange kant naar onder is 0 graden tegen klok in +
                print(f"{center_x},{center_y}")
                
                """so naar ilmar voor deze"""
                dy = center_y_longest_side - center_y
                dx = center_x_longest_side - center_x
                rad = math.atan2(dy, dx)
                angle = math.degrees(rad)
        
            else:
                print("Geen duidelijke hoekpunten gedetecteerd voor driehoek.")
            if area > 50000:
                if R > 100:
                    print("oranje")
                else:
                    print("groen")
            if area < 20000:
                if R < 150: 
                    print("blauw")
                else:
                    print("wit")
            if area > 25000 and area < 29000:
                print("rood")
                eind_hoek = 225
                draai = eind_hoek - angle
                draai = draai - 180
#                if draai < 0:
 #                   pre_draai links
#                else: 
#                    pre_draai rechts
                
        elif num_vertices == 4:
            shape_label = "vierhoek"
            largest_contour = max(contours, key=cv2.contourArea)
            x_as, y_as, w_as, h_as = cv2.boundingRect(largest_contour)
            center_x = x_as + w_as // 2
            center_y = y_as + h_as // 2
            cv2.circle(img_show, (center_x, center_y ), 5, (0, 0, 255), -1)
            rect = cv2.minAreaRect(contour)
            if area > 27000:
                print("geel")
                width = int(rect[1][0])
                height = int(rect[1][1])
                angle = int(rect[2])
                if 0.95 < width / height < 1.05:
                    angle = (angle + 90) % 90
            else:
                print("paars")
                width = int(rect[1][0])
                height = int(rect[1][1])
                angle = int(rect[2])
                if width / height < 0.95 or width / height > 1.05:
                   if width < height:
                     angle = 90 - angle
                   else:
                     angle = -angle

        x, y, _, _ = cv2.boundingRect(contour)
        cv2.putText(img_show, shape_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

print(angle)

cv2.imshow("beeld1", img_zoomed_in)
cv2.imshow("beeld2", img_show)
cv2.imshow("beeld3", img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
