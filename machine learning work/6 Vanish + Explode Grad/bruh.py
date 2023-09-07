import cv2 
import numpy as np

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output
    
def convolve2(image, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype = "float32")
    return output

arr = np.array([[-1]*9, 
                [-1, 1, -1, -1, -1,  -1, -1, 1, -1],
                [-1, -1, 1, -1, -1, -1, 1, -1, -1],
                [-1, -1, -1, 1, -1, 1, -1, -1, -1],
                [-1, -1, -1, -1, 1, -1, -1, -1, -1],
                [-1, -1, -1, 1, -1, 1, -1, -1, -1],
                [-1, -1, 1, -1, -1, -1, 1, -1, -1],
                [-1, 1, -1, -1, -1,  -1, -1, 1, -1],
                [-1]*9, ])

kernel1 = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
kernel2 = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
kernel3 = np.array([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]])
print(arr)
new_arr = convolve2D(arr, kernel2)
print(new_arr)