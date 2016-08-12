拷贝 t3\0.png
然后生成比 0 更深 5mm 的 1.png
结果： 完全不收敛, stepSz=1e-6 却不动, 是因为中心对称吗?

c+=5
cv2.imwrite('srcPlanarBR.png', c)
