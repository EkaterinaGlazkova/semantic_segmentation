import cv2

img = cv2.imread("image.jpg")
gauss_blur = cv2.GaussianBlur(img, (5, 5), 0)
median_blur = cv2.medianBlur(gauss_blur, 5)

rows, cols = img.shape[:2]
matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
res = cv2.warpAffine(median_blur, matrix, (cols, rows))

cv2.imshow("image", res)
cv2.imwrite("result.jpg", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
