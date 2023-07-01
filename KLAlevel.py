import json
import cv2
import numpy as np
import csv
import pandas as pd
from collections import Counter
"""def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse"""

with open('Level_1_Input_Data/input.json','r') as f:
    data=json.load(f)
#print(data)


img1=cv2.imread("Level_2_Input_Data/wafer_image_1.png")
img2=cv2.imread("Level_2_Input_Data/wafer_image_2.png")
img3=cv2.imread("Level_2_Input_Data/wafer_image_3.png")
img4=cv2.imread("Level_2_Input_Data/wafer_image_4.png")
img5=cv2.imread("Level_2_Input_Data/wafer_image_5.png")
img6=cv2.imread("Level_2_Input_Data/wafer_image_6.png")
img7=cv2.imread("Level_2_Input_Data/wafer_image_7.png")
img8=cv2.imread("Level_2_Input_Data/wafer_image_8.png")
img9=cv2.imread("Level_2_Input_Data/wafer_image_9.png")
img10=cv2.imread("Level_2_Input_Data/wafer_image_10.png")
img11=cv2.imread("Level_2_Input_Data/wafer_image_11.png")
img12=cv2.imread("Level_2_Input_Data/wafer_image_12.png")
img13=cv2.imread("Level_2_Input_Data/wafer_image_13.png")
img14=cv2.imread("Level_2_Input_Data/wafer_image_14.png")
img15=cv2.imread("Level_2_Input_Data/wafer_image_15.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
img5=cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
img7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
img9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
img10=cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
img11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
img12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
img13 = cv2.cvtColor(img13, cv2.COLOR_BGR2GRAY)
img14 = cv2.cvtColor(img14, cv2.COLOR_BGR2GRAY)
img15=cv2.cvtColor(img15, cv2.COLOR_BGR2GRAY)
images=[img1,img2,img3,img4,img5,img6,img7,img8,img9,img10,img11,img12,img13,img14,img15]



l1=[]

for i in range(len(img1)):
    for j in  range(len(img1[i])):
        if(img1[i][j]!=128 and img1[i][j]!=255):
            l1.append([1,j,599-i])

for i in range(len(img2)):
    for j in  range(len(img2[i])):
        if(img2[i][j]!=255 and img2[i][j]!=128):
            l1.append([2,j,599-i])
for i in range(len(img3)):
    for j in  range(len(img3[i])):
        if(img3[i][j]!=255 and img3[i][j]!=128):
            l1.append([3,j,599-i])
for i in range(len(img4)):
    for j in  range(len(img4[i])):
        if(img4[i][j]!=255 and img4[i][j]!=128):
            l1.append([4,j,599-i])
for i in range(len(img5)):
    for j in  range(len(img5[i])):
        if(img5[i][j]!=255 and img5[i][j]!=128):
            l1.append([5,j,599-i])
        
with open ('defectdie.csv','w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(l1)

df=pd.read_csv('defectdie.csv',header=None)
df=(df.to_string(index=False, header=False))
l2=[]
lis=[]
dict={}
for i in range(len(images[0])):
    for j in range(999):
        lis=[images[0][i][j],images[1][i][j],images[2][i][j],images[3][i][j],images[4][i][j],images[5][i][j],images[6][i][j],images[7][i][j],images[8][i][j],images[9][i][j],images[10][i][j],images[11][i][j],images[12][i][j],images[13][i][j],images[14][i][j]]
        x=Counter(lis)
        
        key=list(x.keys())
        val=list(x.values())
        maxi=max(val)
        indexx=val.index(maxi)
        max_key=key[indexx]
        for l in range(len(images)):
            
                if(images[l][i][j]!=max_key):
                    l2.append([l+1,j,999-i])
print(l2)
with open ('defect2die.csv','w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(l2)

df=pd.read_csv('defect2die.csv',header=None)
df=(df.to_string(index=False, header=False))
