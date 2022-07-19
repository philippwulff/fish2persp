import cv2
import numpy as np
import sys

assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3, 'OpenCV version 3 or newer required.'

DIM=(1920, 1080)
K=np.array([[380.80522028797867, 0.0, 812.9847908278242], [0.0, 379.05655167042704, 620.923227614725], [0.0, 0.0, 1.0]])
D=np.array([[0.07975586977242274], [-0.028038108388550146], [0.012484750911124085], [-0.004575002647778063]])

K=np.array([[385.4722758605197, 0.0, 814.6418218312826], [0.0, 382.06456852006403, 623.7174971565858], [0.0, 0.0, 1.0]])
D=np.array([[0.049779632806243417], [0.05759876711008945], [-0.1194288078148974], [0.05590870046388291]])

# use Knew to scale the output
Knew = K.copy()
Knew[(0,1), (0,1)] = 0.4 * Knew[(0,1), (0,1)]

p = 'calib_files/3.jpg'

# img = cv2.imread(p)
# img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
# # cv2.imwrite('fisheye_sample_undistorted.jpg', img_undistorted)
# cv2.imshow('undistorted', img_undistorted)
# cv2.waitKey()

def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def undistort2(img_path):
    img = cv2.imread(img_path)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    nemImg = cv2.remap( img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    nk = K.copy()
    nk[0,0]=K[0,0]/2
    nk[1,1]=K[1,1]/2
    # Just by scaling the matrix coefficients!

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), nk, DIM, cv2.CV_16SC2)  # Pass k in 1st parameter, nk in 4th parameter
    nemImg = cv2.remap( img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", nemImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def undistort_all(img_path, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remap(p):
    image_gray = cv2.imread(p)
    img_dim_out = image_gray.shape[:-1]

    K_new = K

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, img_dim_out, cv2.CV_32FC1)
    print("Rectify Map1 Dimension:\n", map1.shape)
    print("Rectify Map2 Dimension:\n", map2.shape)

    undistorted_image_gray = cv2.remap(image_gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # fig = plt.figure()
    # plt.imshow(undistorted_image_gray, "gray")
    
    ret, corners = cv2.findChessboardCorners(image_gray, (6,8),cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    corners_subpix = cv2.cornerSubPix(image_gray, corners, (3,3), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    undistorted_corners = cv2.fisheye.undistortPoints(corners_subpix, K, D)
    undistorted_corners = undistorted_corners.reshape(-1,2)


    fx = K_new[0,0]
    fy = K_new[1,1]
    cx = K_new[0,2]
    cy = K_new[1,2]
    undistorted_corners_pixel = np.zeros_like(undistorted_corners)

    for i, (x, y) in enumerate(undistorted_corners):
        px = x*fx + cx
        py = y*fy + cy
        undistorted_corners_pixel[i,0] = px
        undistorted_corners_pixel[i,1] = py
        
    undistorted_image_show = cv2.cvtColor(undistorted_image_gray, cv2.COLOR_GRAY2BGR)
    for corner in undistorted_corners_pixel:
        image_corners = cv2.circle(np.zeros_like(undistorted_image_show), (int(corner[0]),int(corner[1])), 15, [0, 255, 0], -1)
        undistorted_image_show = cv2.add(undistorted_image_show, image_corners)

    cv2.imshow("undistorted", undistorted_image_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


undistort_all(p, .9)