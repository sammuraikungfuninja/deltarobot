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
temp = 0 
ir = 0
foto = 0

def read_bool(db_number, start_offset, bit_offset):  # To read 1 bit from a specific variable in a DB
    reading = plc.db_read(db_number, start_offset, 1)
    a = snap7.util.get_bool(reading, 0, bit_offset)
    # print('DB Number: ' + str(db_number) + ' Bit: ' + str(start_offset) + '.' + str(bit_offset) + ' Value: ' + str(a))
    return a

def write_lreal_db(db_number, start_address, value):  # To write 1 Lreal to a specific variable in a DB
    plc.db_write(db_number, start_address, bytearray(struct.pack('>d', value)))  # big-endian
    return None

def write_bool(db_number, start_offset, bit_offset, value):  # To write 1 bit to a specific variable in a DB
    reading = plc.db_read(db_number, start_offset, 1)  # (db number, start offset, read 1 byte)
    snap7.util.set_bool(reading, 0, bit_offset, value)  # (value 1= true;0=false) (bytearray_: bytearray, byte_index: int, bool_index: int, value: bool)
    plc.db_write(db_number, start_offset, reading)  # write back the bytearray and now the boolean value is changed in the PLC.
    return None

height = 1080
width = 1920
cv2.namedWindow("beeld1",cv2.WINDOW_AUTOSIZE)
            
cv2.moveWindow("beeld1",0,0) 
webcam = cv2.VideoCapture(0) 
            
webcam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,height) 

groen = 0
oranje = 0
wit = 0
blauw = 0 
rood = 0
geel = 0
paars = 0

while 1:
    retval, img = webcam.read()
    
    if(retval == True):
        cv2.imshow("beeld1",img) 
        cv2.waitKey(1)  
        
    ir = read_bool(18, 18, 3)
#    print(ir)
    if ir == 0:
        temp = 0
    if temp == 0:
       foto = 0
       write_bool(18, 0, 0, foto)
       groen = 0
       oranje = 0
       wit = 0
       blauw = 0 
       rood = 0
       geel = 0
       paars = 0
       write_bool(18, 0, 0, foto)
       write_bool(18, 18, 1, groen)
       write_bool(18, 18, 5, oranje)
       write_bool(18, 18, 4, wit)
       write_bool(18, 18, 7, blauw)
       write_bool(18, 18, 6, rood)
       write_bool(18, 19, 1, geel)
       write_bool(18, 19, 0, paars)
       if ir == 1: 
            for i in range(15):
                retval, img = webcam.read()
                if(retval == True):
                    cv2.imshow("beeld1",img) 
                    cv2.waitKey(1) 
            cv2.imwrite("afbeelding.jpg",img)
            temp = 1
            

            img = cv2.imread(r'C:\Users\daanb\Documents\python\afbeelding.jpg')
            
            x, y, width, height = 690, 415, 400, 350
            roi = img[y:y+height, x:x+width]
            img_zoomed_in = cv2.resize(roi, (width * 2, height * 2))
            
            cv2.imshow("inzoom", img_zoomed_in)
            
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
            robot_x = 0
            robot_y = 0
            angle = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                print(area)
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
                            
                            robot_x = center_x*-0.094 + 0.21*center_y -175.97
                            robot_y = center_x*0.19 + center_y*0.14 + 2.99
                            robot_z = 0
                            if robot_x < -180 and robot_x > -100 and robot_y < 50 and robot_y > 160:
                                print("daan is restarded")
                            else:
                                if center_x < 202:
                                    robot_z = -524
                                elif center_x > 544:
                                    robot_z = -514
                                else:
                                    robot_z = -516
                                write_lreal_db(18,20, robot_z)  
                                write_lreal_db(18, 2, robot_x)
                                write_lreal_db(18, 10, robot_y) 
                                print(f"{robot_x}, {robot_y}")
                            dy = center_y_longest_side - center_y
                            dx = center_x_longest_side - center_x
                            rad = math.atan2(dy, dx)
                            angle = math.degrees(rad)
                        else:
                            print("Geen duidelijke hoekpunten gedetecteerd voor driehoek.")
                        if area > 50000:
                            if R > 100:
                                print("oranje")
                                oranje = 1
                            else:
                                print("groen")
                                groen = 1
                        if area < 20000:
                            if R < 150: 
                                print("blauw")
                                blauw = 1
                            else:
                                print("wit")
                                wit = 1
                        if area > 25000 and area < 40000:
                            print("rood")
                            rood = 1
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
                        robot_x = center_x*-0.094 + 0.21*center_y -175.97
                        robot_y = center_x*0.19 + center_y*0.14 + 2.99
                        robot_z = 0
                        if robot_x < -180 and robot_x > -100 and robot_y < 50 and robot_y > 160:
                            print("daan is restarded")
                        else:
                            if center_x < 202:
                                robot_z = -524
                            elif center_x > 544:
                                robot_z = -514
                            else:
                                robot_z = -516
                            write_lreal_db(18,20, robot_z)  
                            write_lreal_db(18, 2, robot_x)
                            write_lreal_db(18, 10, robot_y) 
                            print(f"{robot_x}, {robot_y}")
                        if area > 34000:
                            print("geel")
                            geel = 1
                            width = int(rect[1][0])
                            height = int(rect[1][1])
                            angle = int(rect[2])
                            if 0.95 < width / height < 1.05:
                                angle = (angle + 90) % 90
                        else:
                            print("paars")
                            paars = 1
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
            foto = 1
            write_bool(18, 0, 0, foto)
            write_bool(18, 18, 1, groen)
            write_bool(18, 18, 5, oranje)
            write_bool(18, 18, 4, wit)
            write_bool(18, 18, 7, blauw)
            write_bool(18, 18, 6, rood)
            write_bool(18, 19, 1, geel)
            write_bool(18, 19, 0, paars)
           # cv2.imshow("zoomed_in", img_zoomed_in)
            cv2.imshow("contour_middelpunt", img_show)
            cv2.imshow("binary", img_binary)
          #  cv2.waitKey(1)
          #  cv2.destroyAllWindows()
