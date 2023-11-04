import cv2
import numpy as np
import matplotlib.pyplot as plt

def to_find_TKS(input_image_path):
    # Read the input image
    input_image = cv2.imread(input_image_path)

    # Check if the input image is loaded correctly
    if input_image is None:
        print("Error: Check if the input image path is correct")
        return

    # Convert the input image to grayscale
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to determine the optimal threshold value
    threshold_value, tup = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine the optimal Gaussian blur kernel size and standard deviation based on the image size
    height, width = grayscale_image.shape
    # Ensure kernel size is odd and at least 3
    kernel_size = (max(3, int(width / 10) | 1), max(3, int(height / 10) | 1))
    sigma = (kernel_size[1]*0.5 - 1)*0.2

    return threshold_value, kernel_size, sigma

def perform_bokesh(input_image_path, output_image_path):
    # Read the input image
    input_image = cv2.imread(input_image_path)

    # Check if the input image is loaded correctly
    if input_image is None:
        print("Error: Check if the input image path is correct")
        return

    # Determine parameters based on the input image
    threshold_value, kernel_size, sigma = to_find_TKS(input_image_path)

    # Convert the input image to grayscale
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding using the determined threshold value
    _, binary_mask = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Invert the binary mask
    inverted_mask = cv2.bitwise_not(binary_mask)

    # Apply Gaussian blur to the background using the determined kernel size and sigma
    blurred_background = cv2.GaussianBlur(input_image, kernel_size, sigma)

    # Combine the segmented subject and blurred background
    input_image[inverted_mask == 255] = blurred_background[inverted_mask == 255]

    # Save or display the final image
    cv2.imwrite(output_image_path, input_image)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.title('Blurred')
    plt.show()

# Input and output image paths
input_image_path = 'image2.jpg'
output_image_path = 'output.jpg'

# Call the function to segment and blur the background
perform_bokesh(input_image_path, output_image_path)