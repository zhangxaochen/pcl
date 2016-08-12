正常无 reset

#eg2.
d=arange(1000, 1640, dtype=uint16)
e=tile(d, (480,1))
e[:200, :300]=0 #左上方块置零
cv2.imshow('ttt', e)
cv2.imwrite('srcPlanarBR.png', e)
