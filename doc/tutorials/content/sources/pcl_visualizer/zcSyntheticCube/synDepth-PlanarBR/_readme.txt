zhangxaochen:	//2016-6-27 10:49:14
构造 位于屏幕BottomRight 的， 平行于成像平面的深度图
作为《深度图配准方案实验结果》测试用例

#---------------py代码:
%pylab
import cv2
c=zeros((480,640), dtype=uint16)
c[280:-40,360:-40]=1000
cv2.imshow('ttt', c)
cv2.imwrite('srcPlanarBR.png', c)
c[280:-40,360:-40]=1005
cv2.imwrite('dstPlanarBR.png', c)

#eg2.
d=arange(1000, 1640, dtype=uint16)
e=tile(d, (480,1))
e[:200, :300]=0 #左上方块置零
cv2.imshow('ttt', e)

