# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:02:32 2019

@author: retdu
"""

import os
import cv2

path="G:/untitled5/static/1/"
target="G:/untitled5/static/9/"
for item in os.listdir(path):
    img=cv2.imread(path+item)
    img=cv2.resize(img,(100,100))
    #img = cv2.resize(img,(100,100))
    b=cv2.blur(img,(3,3))
    
    cv2.imwrite(target+item,b)