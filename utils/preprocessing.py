import cv2
import numpy as np

def skew_correction(gray_image):
    orig = gray_image

    # Threshold to get rid of extraneous noise using Otsu's thresholding
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Blur the image
    blur = cv2.GaussianBlur(thresh, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Perform probabilistic Hough Line Transform
    # cv2.HoughLinesP returns an array of lines in the format [[x1, y1, x2, y2]]
    hough_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    slopes = []
    if hough_lines is not None:
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                slopes.append(slope)
            else:
                slopes.append(float('inf')) # Vertical lines

    # Calculate angles from slopes
    rad_angles = [np.arctan(x) for x in slopes if x != float('inf') and x != -float('inf')]
    # Convert to degrees for rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # Find the most common angle for rotation
    if deg_angles:
        histo = np.histogram(deg_angles, bins=100)
        rotation_number = histo[1][np.argmax(histo[0])]

        # Correcting for 'sideways' alignments
        if rotation_number > 45:
            rotation_number = -(90 - rotation_number)
        elif rotation_number < -45:
            rotation_number = 90 - abs(rotation_number)
    else:
        rotation_number = 0 # No lines found, no rotation

    # Rotate the image to deskew it
    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, rotation_number, 1.0)
    rotated = cv2.warpAffine(orig, matrix, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, rotation_number



def resize_image(rotated_image):

    # Get original dimensions
    (h, w) = rotated_image.shape[:2]
    max_dim = 640

    # Calculate the ratio to scale down the image
    if w > h:
        ratio = max_dim / float(w)
        new_w = max_dim
        new_h = int(h * ratio)
    else:
        ratio = max_dim / float(h)
        new_h = max_dim
        new_w = int(w * ratio)

    # Resize the image
    scaled_img = cv2.resize(rotated_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    print(f"Original dimensions: ({w}x{h})")
    print(f"Scaled dimensions: ({new_w}x{new_h})")
    return scaled_img



def add_border(scaled_img):
    border_image = cv2.copyMakeBorder(src=scaled_img, top=20, bottom=20, left=20, right=20, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return border_image




