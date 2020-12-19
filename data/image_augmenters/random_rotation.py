from image_augmenters.bbox_utility import clip_box, get_corners, rotate_box, get_enclosing_box, rotate_im
import numpy as np
import random
import cv2

class RandomRotation(object):
    """Randomly Rotates the image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Usage
    ----------
    rotate = RandomRotate(20)
    img, bboxes = rotate(img, bboxes)
    plt.imshow(draw_rect(img, bboxes))


    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Translated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """
    def __init__(self, angle = 10):
        self.angle = angle

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"
        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, bboxes):

        angle = random.uniform(*self.angle)

        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
        img = rotate_im(img, angle)

        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:,4:]))
        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
        new_bbox = get_enclosing_box(corners)

        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h
        img = cv2.resize(img, (w,h))

        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
        bboxes  = new_bbox
        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)

        return img, bboxes
