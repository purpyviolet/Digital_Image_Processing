import os
import cv2
import numpy as np

#读取图片和缩放图片
pic1=cv2.imread('high_res.png')
pic1=cv2.resize(src=pic1,dsize=(450,450))
#创建相同尺寸的图片
npimg=np.ones(shape=(pic1.shape[0],pic1.shape[1],pic1.shape[2]),dtype=np.uint8)*100

#两张图片进行与运算
dst=cv2.add(src1=pic1, src2=npimg)
#显示图片
cv2.imshow('pic1',pic1)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

