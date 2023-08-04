import numpy as np
import cv2
scale = 4
# img = ['8023', '78004','86000','86016','108005','253027','291000']
# x1 = [182,180,151,210,353,114,110]
# y1 = [143,115,94,70,88,147,200]
# x2 = [227,212,202,276,400,188,209]
# y2 = [185,152,134,93,161,286]
def dpsr(img, x1, y1, x2, y2):
    for img, x1, y1, x2, y2 in zip(img, x1, y1, x2, y2):
        image_path2 = '/home/lqg/code/results/B100/dpsr/3/'+str(img)+'_x4_.png'
        Simage_path = '/home/lqg/code/results/pic/B100/dpsr/'+str(img)+'S.png'
        image2 = cv2.imread(image_path2)

        SRfirst_point = (x1, y1)
        SRlast_point = (x2, y2)
        cv2.imwrite(Simage_path, image2[y1:y2, x1:x2])

