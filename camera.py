import cv2
import numpy as np
import glob
import math

assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'


class PiCam:

    # {"k":[0.047992,0.020061,-0.017053,0.002015],"f":[0.188320,0.251545],"c":[0.5,0.5],"maxfov":220}

    DIM = (1920, 1080)
    # FOV_DIM = (640, 360)            # on Unity side they want 1920 x 1080 ratio
    FOV_DIM = (400, 400)

    THETA_MAX = 60
    PHI_MAX = 90

    K_LEFT = np.array([[385.4722758605197, 0.0, 814.6418218312826], 
                        [0.0, 382.06456852006403, 623.7174971565858], 
                        [0.0, 0.0, 1.0]])
    D_LEFT = np.array([[0.049779632806243417], [0.05759876711008945], [-0.1194288078148974], [0.05590870046388291]])
    X_OFFSET_LEFT = 0

    K_RIGHT = np.array([[385.4722758605197, 0.0, 814.6418218312826], 
                        [0.0, 382.06456852006403, 623.7174971565858], 
                        [0.0, 0.0, 1.0]])
    D_RIGHT = np.array([[0.049779632806243417], [0.05759876711008945], [-0.1194288078148974], [0.05590870046388291]])
    X_OFFSET_RIGHT = 120

    # # 1
    # K_RIGHT = np.array([[373.53971442261695, 0.0, 953.0065981390961], [0.0, 371.6175021081148, 530.5431853370384], [0.0, 0.0, 1.0]])
    # D_RIGHT = np.array([[0.08181846680434758], [-0.0037859783556971655], [-0.014526325833358314], [0.003678459415661634]])
    # # 2
    # K_RIGHT = np.array([[383.58525659395895, 0.0, 956.628287861117], [0.0, 382.1446767721286, 533.4155001657831], [0.0, 0.0, 1.0]])
    # D_RIGHT = np.array([[0.050825574642767844], [0.022269358682706082], [-0.020574134910108215], [0.002824074658537577]])
    # # 3
    # K_RIGHT = np.array([[284.0220886491106, 0.0, 954.642229319893], [0.0, 281.92904786890597, 531.0240672726558], [0.0, 0.0, 1.0]])
    # D_RIGHT = np.array([[0.17051237875939154], [0.04370205451898174], [-0.0452797300173468], [0.01024286315880872]])

    def __init__(self, stereo=False, mono_cam="left"):
        self.stereo = stereo
        self.mono_cam = mono_cam
        self.cam_left = None
        self.cam_right = None

    def start_cameras(self):
        if self.stereo:
            self.cam_left = cv2.VideoCapture(0)
            self.cam_right = cv2.VideoCapture(1)
        else:
            if self.mono_cam == "left":
                self.cam_left = cv2.VideoCapture(0)
            else:
                self.cam_right = cv2.VideoCapture(0)

    def kill_cameras(self):
        self.cam_left.release()
        self.cam_right.release()

    def read(self, angles, angle_type="spherical", draw_bb=False):
        head_x, head_y, head_z = None, None, None
        if angle_type == "euler":
            head_x, head_y, head_z = angles
        elif angle_type == "spherical":
            theta, phi = angles
        else:
            raise NotImplementedError

        frame_l, frame_r, frame_lp, frame_rp, frame_lu, frame_ru = [None] * 6
        if self.cam_left:
            ret, frame_l = self.cam_left.read()
        if self.cam_right:
            ret, frame_r = self.cam_right.read()

        if theta is None and phi is None:
            theta, phi = self.angles2spherical(head_x, head_y, head_z)

        if frame_l is not None:
            frame_lu = self.undistort(frame_l, self.K_LEFT, self.D_LEFT, self.DIM, balance=0.9)
            bb, center = self.spherical2bb(theta, phi, x_offset=self.X_OFFSET_LEFT)
            if draw_bb:
                cv2.circle(frame_lu, center, radius=1, color=(0, 0, 255), thickness=10)
            frame_lp = self.perspective_transformation(frame_lu, bb, draw_bb=draw_bb)
        if frame_r is not None:
            frame_ru = self.undistort(frame_r, self.K_RIGHT, self.D_RIGHT, self.DIM, balance=0.9)
            bb, center = self.spherical2bb(theta, phi, x_offset=self.X_OFFSET_RIGHT)
            if draw_bb:
                cv2.circle(frame_ru, center, radius=1, color=(0, 0, 255), thickness=10)
            frame_rp = self.perspective_transformation(frame_ru, bb, draw_bb=draw_bb)        
        
        return frame_lp, frame_rp, frame_lu, frame_ru

    def spherical2xy(self, theta, phi, x_offset=0):
        """Helper"""
        W, H = self.DIM
        x = int(W/2 + W/2 * phi/self.PHI_MAX)
        shift = min(H*6//7, int(H/2 + H/2 * theta/self.THETA_MAX))
        y, dy = self.get_y_on_iso(x, W/2 - x_offset)
        y += shift
        return (x, y), dy

    def spherical2bb(self, theta, phi, x_offset=0):
        """
        Right-handed coordinate frame with spherical coordinates. 
        The camera points in the Y-direction and rotates about the Z-axis.
        """
        theta = self.clamp(theta, -self.THETA_MAX, self.THETA_MAX)
        phi = self.clamp(phi, -self.PHI_MAX, self.PHI_MAX)

        center, dy = self.spherical2xy(theta, phi, x_offset=x_offset)
        angle = np.arctan(dy) * 180/np.pi # in degrees
        M = cv2.getRotationMatrix2D(center, angle, 1)
        w, h = self.FOV_DIM
        # top_left, top_right, bottom_right, bottom_left
        bb = np.array([(0, 0), (w, 0), (w, h), (0, h)]) + np.array(center) - np.array([w/2, h/2])
        bb = [self.rotate((center), p, angle * np.pi/180) for p in bb]
        
        return np.array(bb).astype(int), center

    @staticmethod
    def rotate(origin, point, angle):
        """Rotate a point counterclockwise by a given angle around a given origin. Angle is in radians."""
        ox, oy = origin
        px, py = point
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return int(qx), int(qy)

    @staticmethod
    def perspective_transformation(img, bb, draw_bb=False):
        """From https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/"""
        rect = cv2.minAreaRect(bb)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if draw_bb:
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        #width = int(rect[1][0])
        #height = int(rect[1][1])
        width = int(np.linalg.norm(bb[0]-bb[1]))
        height = int(np.linalg.norm(bb[0]-bb[3]))

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (width, height))
        if bb[0, 1] < bb[1, 1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return warped
    

    @staticmethod
    def angles2spherical(head_x, head_y, head_z):
        phi = head_y
        theta = (head_z + head_x)/2
        return theta, phi

    @staticmethod
    def clamp(n, smallest, largest): 
        return max(smallest, min(n, largest))

    @staticmethod
    def get_y_on_iso(x, center_x, lin_m=0.4):
        x -= center_x       # center x coordinate
        x_meet = center_x/3    # where linear meets quadratic
        if - x_meet <= x <= x_meet:
            c0 = lin_m / (2*x_meet)
            c1 = lin_m * x_meet - c0 * x_meet**2
            y = c0 * x**2 +c1
            dy = 2 * c0 * x
        else:
            y = lin_m * abs(x)
            dy = lin_m if x > 0 else -lin_m 
        return int(y), dy

    @staticmethod
    def undistort(img, K, D, DIM, balance=0.0, dim2=None, dim3=None):
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
        return undistorted_img


if __name__ == "__main__":

    picam = PiCam(mono_cam="right")
    picam.start_cameras()
    # _, frame = picam.read()
    phi, theta = 0, 0

    while(True):
        frame_l, frame_r, frame_lall, frame_rall = picam.read((theta, phi), angle_type="spherical", draw_bb=True)

        # W, H = picam.DIM

        # for t in range(-40, 40, 2):
        #     x_, y_ = 0, 1080
        #     for x in range(0, 1920, 20):
        #         shift = min(H*6//7, int(H/2 + H/2 * t/40))
        #         y, dy = picam.get_y_on_iso(x, 1920/2)
        #         y += shift
        #         cv2.line(frame_r, (x_, y_), (x, y), (0, 255, 0), 3)
        #         x_, y_ = x, y

        # bb, center = picam.spherical2bb(theta, phi)
        # cv2.circle(frame_r, center, radius=1, color=(0, 0, 255), thickness=10)
        # frame_c = picam.perspective_transformation(frame_r, bb, draw_bb=True)

        # cv2.imshow('PiCam all', frame_r)

        #cv2.imshow('PiCam Left', frame_l)
        cv2.imshow('PiCam Right', frame_r)
        cv2.imshow('PiCam all', frame_rall)


        key = cv2.waitKey(1) & 0xFF

        if key == 27:       # ESC
            break
        if key == 0: 
            theta -= 1
        if key == 1:
            theta += 1
        if key == 2:
            phi -= 1
        if key == 3:
            phi += 1

    cv2.destroyAllWindows()

