import cv2
import matplotlib.pyplot as plt

img = cv2.imread('data/bolt.jpg',0)
'''
imgTh = cv2.threshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11)
'''
#cv2.imshow('res',img)

#img = cv2.medianBlur(img,5)
laplacian = cv2.Laplacian(img,cv2.CV_64F,ksize=7)
plt.imshow(img, cmap='gray')
plt.show()
plt.imshow(laplacian, cmap='gray')
plt.show()

canny = cv2.Canny(img,100,200)
plt.imshow(canny, cmap='gray')
plt.show()