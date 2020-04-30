
from autoaugment import autoaugdet
import cv2
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

def aug():

    image = cv2.imread("000000011051.jpg")
    bboxes = [  [195.45,197.97,61.16,226.63],
                [249.33,77.09,385.44,458.91],
                [7.23,2.41,386.64,525.16]
             ]
    to_show_image = image.copy()


    autoaugdet.autoaugdet(image, bboxes, "v2")

    return


    for box in bboxes:
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        #cv2.rectangle(to_show_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(255, 0, 0))
    #cv2.imshow("in", to_show_image)
    # cv2.waitKey(0)

    h,w,c = image.shape
    bboxes = np.array(bboxes)
    temp = bboxes[:, 0] / w
    bboxes[:, 0] =  bboxes[:, 1] / h
    bboxes[:, 1] = temp
    temp = bboxes[:, 2] / w
    bboxes[:, 2] = bboxes[:, 3] / h
    bboxes[:, 3] = temp
    # bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0])/ w
    # bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1])/ h

    image, bboxes = autoaugdet.distort_image_with_autoaugment(tf.convert_to_tensor(image), tf.to_float(tf.convert_to_tensor(bboxes)), "v1")
    image = image.numpy()
    bboxes = bboxes.numpy()

    temp = bboxes[:, 0] * h
    bboxes[:, 0] = bboxes[:, 1] * w
    bboxes[:, 1] = temp
    temp = bboxes[:, 2] * h
    bboxes[:, 2] = bboxes[:, 3] * w
    bboxes[:, 3] = temp
    #for i in range(bboxes.shape[0]):
    #    cv2.rectangle(image, (int(bboxes[i, 0]), int(bboxes[i, 1])), (int(bboxes[i, 2]), int(bboxes[i, 3])),(0, 0, 255))
    #cv2.imshow("out", image)
    #cv2.waitKey(1)

if __name__ == "__main__":
    for i in range(1000):
        aug()
        print("Test", i)
