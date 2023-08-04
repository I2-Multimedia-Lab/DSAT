import result_compare as our
import result_compare_dasr as dasr
import result_compare_dpsr as dpsr
import result_compare_srmdnf as srmdnf
import result_compare_swinir as swinir
img = ['8023', '78004','86000','86016','108005','253027','291000']
x1 = [182,180,151,210,353,114,110]
y1 = [143,115,94,70,88,147,200]
x2 = [227,212,202,276,400,188,209]
y2 = [185,152,134,93,161,286]
method = ['our','dasr','dpsr','srmdnf','swinir']
for met in method:
    if met == 'our':
        our.our(img, x1, y1, x2, y2)
        print(met+' over!')
    elif met == 'dasr':
        dasr.dasr(img, x1, y1, x2, y2)
        print(met+' over!')
    elif met == 'dpsr':
        dpsr.dpsr(img, x1, y1, x2, y2)
        print(met+' over!')
    elif met == 'srmdnf':
        srmdnf.srmdnf(img, x1, y1, x2, y2)
        print(met+' over!')
    elif met == 'swinir':
        swinir.swinir(img, x1, y1, x2, y2)
        print(met+' over!')
