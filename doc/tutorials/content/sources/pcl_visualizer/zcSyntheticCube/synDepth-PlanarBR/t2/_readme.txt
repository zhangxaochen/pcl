������ reset

#eg2.
d=arange(1000, 1640, dtype=uint16)
e=tile(d, (480,1))
e[:200, :300]=0 #���Ϸ�������
cv2.imshow('ttt', e)
cv2.imwrite('srcPlanarBR.png', e)
