仍是右下角平面, 但其中内部增加 1200mm 小凸起
#0.png
import cv2
c=zeros((480,640), dtype=uint16)
c[280:-40,360:-40]=1000
c[330:-90,460:-140]=1200
cv2.imshow('ttt', c)

#1.png 增加 5mm
c[c!=0]+=5
