# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:20:29 2020

@author: MAYANK
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import os




files = []
output = []
i = 0


for i in range(1, 1005):
    img = Image.open('./cats/'+str(i)+'.jpg')
    
    new_img = img.resize((64, 64))
    plt.imshow(new_img)
    if new_img.mode != 'RGB':
        new_img = new_img.convert('RGB')
    new_img.save('./catstest/'+str(i)+'.jpg')
    
for i in range(1, 1005):
    img = Image.open('./dogs/'+str(i)+'.jpg')
    
    new_img = img.resize((64, 64))
    plt.imshow(new_img)
    if new_img.mode != 'RGB':
        new_img = new_img.convert('RGB')
    new_img.save('./dogstest/'+str(i)+'.jpg')
    
for i in range(1, 1001, 10):
    for j in range(i, i+10):
    
        img = Image.open('./dogstest/'+str(j)+'.jpg')
        output.append(0)
        files.append(np.asarray(img))
        
    for j in range(i, i+10):
    
        img = Image.open('./catstest/'+str(j)+'.jpg')
        output.append(1)
        files.append(np.asarray(img))
       




        
      
filesoned = []
for file in files:
    print(file.shape)
    filesoned.append(file.reshape(-1))



filesoned = np.array(filesoned).T



np.save('input_data', filesoned)

np.save('output_data', np.array(output))

