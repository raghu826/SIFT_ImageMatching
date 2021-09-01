import cv2 as cv

# reading and resize the first input images
image_1 = cv.imread("Images/gate.jpg")
image_1 = cv.resize(image_1, (512, 512), interpolation=cv.INTER_AREA)
image_2 = cv.imread("Images/arch.jpg")
image_2 = cv.resize(image_2, (512, 512), interpolation=cv.INTER_AREA)

# Convert images into gray scale
gray_1 = cv.cvtColor(image_1, cv.COLOR_BGR2GRAY)
gray_2 = cv.cvtColor(image_2, cv.COLOR_BGR2GRAY)

# SIFT technique
sift = cv.xfeatures2d.SIFT_create(nfeatures=500)

# Exteacting Keypoints and descriptors
kp_2, desc_2 = sift.detectAndCompute(gray_2, None)
kp_1, desc_1 = sift.detectAndCompute(gray_1, None)
print(f'shape of descriptor for Image 1: {desc_1.shape}')

# Displaying the keypoints of Images
image1_keypoints = cv.drawKeypoints(gray_1, kp_1, image_1)
cv.imshow("image1-points", image1_keypoints)
cv.imwrite("Images/keypoints_gate.jpg", image1_keypoints)
image2_keypoints = cv.drawKeypoints(gray_2, kp_2, image_2)
cv.imshow("image2-points", image2_keypoints)
cv.imwrite("Images/keypoints_arch.jpg", image2_keypoints)

# IMAGE MATCHING with brute force
bf = cv.BFMatcher(cv.NORM_L1, crossCheck = True)
matches = bf.match(desc_1, desc_2)
matches = sorted(matches, key=lambda x:x.distance)

final_img = cv.drawMatches(image_1, kp_1, image_2, kp_2, matches[:50],image_2, flags=2)
cv.imshow('matching', final_img)
cv.imwrite("Images/ImageMatching.jpg", final_img)
cv.waitKey(0)

