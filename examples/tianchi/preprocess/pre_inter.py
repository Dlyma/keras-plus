# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as itp

# written by zhaowuxia @ 2015/6/8
# used for generate datasets for the tianchi
def generate(csv_path, interest_path):
    x = []
    y = []
    xval = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        i = 0
        for line in lines[1:]:
            line = line.strip().split(',')
            try:
                interest_O_N = float(line[3])
                interest_1_W = float(line[4])
                interest_2_W = float(line[5])
                interest_1_M = float(line[6])
                interest_3_M = float(line[7])
                interest_6_M = float(line[8])
                interest_9_M = float(line[9])
                interest_1_Y = float(line[10])
                x.append(i)
                y.append([interest_O_N, interest_1_W, interest_2_W, interest_1_M, interest_3_M, interest_6_M, \
                        interest_9_M, interest_1_Y])
            except:
                xval.append(i)
            finally:
                i += 1
        x.append(i)
        y.append([2.9060, 3.5840, 4.1170, 4.0570, 4.6665, 4.8716, 4.9373, 5.])
    x = np.array(x)
    y = np.array(y)
    xval = np.array(xval)
    yitp = itp.spline(x, y, xval)
    print(x.shape, y.shape, xval.shape, yitp.shape)

    with open(interest_path, 'w') as f:
        j = 0
        k = 0
        for i in range(len(x) -1 + len(xval)):
            if j < len(x) and x[j] == i:
                for v in y[j]:
                    f.write('%.4f,'%v)
                j += 1
            elif k < len(xval) and xval[k] == i:
                for v in yitp[k]:
                    f.write('%.4f,'%v)
                k += 1
            else:
                print('error', i)
            f.write('\n')
    

if __name__=='__main__':
    generate('/home/zhaowuxia/dl_tools/datasets/tianchi/total.csv', 'out.csv')
