import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import perspective, Plane, load_camera_params, bilinear_sampler

image = cv2.cvtColor(cv2.imread('1403774747495925.png'), cv2.COLOR_BGR2RGB)
TARGET_H, TARGET_W = 1024,1024


def ipm_from_parameters(image, xyz, K, RT):
    P = K @ RT
    print(P)
    pixel_coords = perspective(xyz, P, TARGET_H, TARGET_W)
    image2 = bilinear_sampler(image, pixel_coords)
    return image2.astype(np.uint8)



if __name__ == '__main__':
    ################
    # Derived method
    ################
    # Define the plane on the region of interest (road)
    plane = Plane(20, -25, 0, 0, 0, 0, TARGET_H, TARGET_W, 0.035)
    # Retrieve camera parameters
    extrinsic, intrinsic = load_camera_params('camera.json')
    # Apply perspective transformation
    warped1 = ipm_from_parameters(image, plane.xyz, intrinsic, extrinsic)

 

    # Draw results
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(image)
    ax[0].set_title('Front View')
    ax[1].imshow(warped1)
    ax[1].set_title('IPM')
    plt.show()
