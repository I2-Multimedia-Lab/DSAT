import numpy as np
import cv2
scale = 4
img = ['AkkeraKanjinchou','ARMS','Belmondo','BEMADER_P','DollGun','DualJustice','HighschoolKimengumi_vol01','HinagikuKenzan',
'JangiriPonpon','JijiBabaFight','KyokugenCyclone','LoveHina_vol01','MagicianLoad','MariaSamaNihaNaisyo','MayaNoAkaiKutsu',
'MukoukizuNoChonbo','ParaisoRoad','PikaruGenkiDesu','PlatinumJungle','PsychoStaff','Raphael',
'ShimatteIkouze_vol01','TapkunNoTanteisitsu','TensiNoHaneToAkumaNoShippo','TetsuSan','ThatsIzumiko_000','TotteokiNoABC',
'UltraEleven','YumeiroCooking']
x1 = [264,729,448,504,629,487,289,243,135,162,344,387,523,643,224,588,359,62,670,682,11,202,283,18]
y1 = [16,346,15,929,905,420,910,603,684,832,713,68,772,15,53,518,1003,397,917,437,588,248,1,667]
x2 = [541,817,732,625,777,585,403,335,261,329,432,566,645,803,309,808,458,315,772,757,264,462,532,176]
y2 = [64,466,63,1081,961,523,1011,751,807,937,845,170,947,127,357,546,1090,474,1127,535,614,286,47,992]
def our(img, x1, y1, x2, y2):
    for img, x1, y1, x2, y2 in zip(img, x1, y1, x2, y2):
        image_path1 = '/home/lqg/code/results/manga109/hr/'+str(img)+'.png'
        image_path2 = '/home/lqg/code/results/manga109/ours/1/'+str(img)+'_x4.0_SR.png'
        image_path3 = '/home/lqg/code/results/manga109/lr/1/'+str(img)+'_x4_.png'
        Himage_path = '/home/lqg/code/results/pic/manga1092/dsat/'+str(img)+'H.png'
        Simage_path = '/home/lqg/code/results/pic/manga1092/dsat/'+str(img)+'S.png'
        Limage_path = '/home/lqg/code/results/pic/manga1092/dsat/'+str(img)+'L.png'
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        image3 = cv2.imread(image_path3)
        SRfirst_point = (x1, y1)
        SRlast_point = (x2, y2)
        cv2.imwrite(Simage_path, image2[y1:y2, x1:x2])
        LRfirst_point = (int(x1/scale), int(y1/scale))
        LRlast_point = (int(x2/scale), int(y2/scale))
        cv2.rectangle(image3, LRfirst_point, LRlast_point, (0, 0, 255), 2)
        cv2.imwrite(Limage_path, image3)
        HRfirst_point = (x1, y1)
        HRlast_point = (x2, y2)
        cv2.imwrite(Himage_path, image1[y1:y2, x1:x2])

our(img, x1, y1, x2, y2)