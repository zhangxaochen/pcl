四周平面1000, 中间更深矩形 1555

c[:]=1000
c[200:-200,300:-300]=1555
cv2.imshow('ttt', c)
cv2.imwrite('srcPlanarBR.png', c)