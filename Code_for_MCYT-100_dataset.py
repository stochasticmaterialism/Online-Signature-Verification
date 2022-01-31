import os
from matplotlib import pyplot as plt
import numpy as np
import statistics
from scipy.spatial import distance
from numpy.random import uniform
from scipy import interpolate
from scipy.special import eval_hermite
from scipy.signal import hilbert
from scipy.fftpack import dct
from scipy.fftpack import dst
import math
from math import floor
import pandas as pd
from pywt import dwt
'''from PyEMD import EMD'''
from sklearn.metrics import roc_curve
!mkdir "abcabc"
!unzip "0055.zip" -d abcabc

###########################3FEATURE EXTRACTION TRAINING SET###########################################333
# MAXIMUM CALCULATION
maximum = 0
file1 = open(r"/content/abcabc/0000/TRAIN/0000f00.txt", 'r')
datapath = r"/content/abcabc/0055"
data = os.listdir(datapath)
for ab in data:
  p1 = datapath + "/" + ab
  print(p1)
  datad = os.listdir(p1)
  for dataset in datad:
      file1.seek(0)
      path = p1+"/" +dataset
      List_coor_target = []
      k=0
      count = 0
      print(path)
      file1 = open(path, 'r')
      length = 0
      while True:
          d = 0
          add = 0
          n=0
          line = file1.readline()  
          if not line: 
              break
          if(count == 1):
              k = 1;
          count = count + 1
          if(k == 1):
              length = length + 1
              for i in range(len(line)):
                  if(line[i] == " "):
                      d=d+1
                  if(d == 0):
                      temp = i+1
                  if(d==1):
                      temp2 = i+1
              code = line[temp:temp2]
              '''print(code)'''
              code = int(code)
              List_coor_target.append(code)
      if(length >= maximum):
          maximum = length
      print(dataset)
      print(length)
print("MAXIMUM" , maximum)
length_query = maximum

###################################### FEATURE EXTRACTION TRAINING SET--------LOGICAL FEATURES#######################
data_set = []
file1 = open(r"/content/abcabc/0000/TRAIN/0000f00.txt", 'r')
ind = []
List_coor = []
y = []
no_of_img = 0
datapath = r"/content/abcabc/0055/TRAIN"
data = os.listdir(datapath)
for dataset in data:
    print(dataset)
    List_coor_target = []
    List_coor_target_y = []
    List_coor_target_azi = []
    List_coor_target_alt = []
    List_coor_target_p = []
    List_coor_target_t = []
    time = 0
    count = 0
    k = 0
    file1.seek(0)
    path = datapath+"/" +dataset
    no_of_img = no_of_img + 1
    name = dataset
    '''for u in range(len(dataset)):
      if(name[u] == "S"):
        temp = u+1
    number = name[temp:len(dataset)-4]
    number = int(number)'''
    if(name[4] == 'v'):
      print(name[4])
      y.append("False")
      print("False")
    else:
      print(name[4])
      y.append("True")
      print("True")
    ind.append(no_of_img)
    file1 = open(path, 'r')
    while True:
        d = 0
        add = 0
        n=0
        line = file1.readline()  
        if not line: 
            break
        if(count == 1):
            k = 1;
        count = count + 1
        if(k == 1):
          for i in range(len(line)):
              if(line[i] == " "):
                  lead = 0
                  lead1 = 0
                  lead2 = 0
                  lead3 = 0
                  lead4 = 0
                  lead5 = 0
                  d=d+1
              if(d == 1 and lead == 0):
                  temp = i+1
                  lead = 1
              if(d == 2 and lead1 == 0):
                  temp2 = i+1
                  lead1 = 1
              if(d == 4 and lead2 == 0):
                  temp3 = i+1
                  lead2 = 1
              '''if(d == 5 and lead3 == 0):
                  temp4 = i+1
                  lead3 = 1
              if(d == 6 and lead4 == 0):
                  temp5 = i+1
                  lead4 = 1'''
              if(d == 3 and lead5 == 0):
                  temp6 = i+1
                  lead5 = 1
          time = time+1
          code = line[0:temp]
          code_y = line[temp:temp2]
          code_p = line[temp2:temp6]
          code_azi = line[temp6:temp3]
          code_alt = line[temp3:len(line)]
          '''code_p = line[temp5:len(line)]'''
          code = int(code)
          code_y = int(code_y)
          code_p = int(code_p)
          code_azi = int(code_azi)
          code_alt = int(code_alt)
          '''code_p = int(code_p)'''
          List_coor.append([no_of_img , time , code , code_y , code_azi , code_alt , code_p])
          List_coor_target.append(code)
          List_coor_target_y.append(code_y)
          List_coor_target_azi.append(code_azi)
          List_coor_target_alt.append(code_alt)
          List_coor_target_p.append(code_p)
          length_target = len(List_coor_target)

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_x = List_coor_target
    f = interpolate.interp1d(x, y_x)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_x = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target = ynew_x

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_y = List_coor_target_y
    f = interpolate.interp1d(x, y_y)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_y = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_y = ynew_y

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_p = List_coor_target_p
    f = interpolate.interp1d(x, y_p)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_p = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_p = ynew_p

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_alt = List_coor_target_alt
    f = interpolate.interp1d(x, y_alt)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_alt = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_alt = ynew_alt

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_azi = List_coor_target_azi
    f = interpolate.interp1d(x, y_azi)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_azi = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_azi = ynew_azi

    for er in range(len(List_coor_target)):
      List_coor_target_t.append(er+1)
    
    '''x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_t = List_coor_target_t
    f = interpolate.interp1d(x, y_t)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_t = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_t = ynew_t'''
    

    disp = []
    velo_x = []
    velo_y = []
    velocity_x = []
    velocity_y = []
    abs_velo = []
    abs_velo = []
    ac_x = []
    ac_y = []
    abs_ac = []
    cent_ac = []
    cos_alpha = []
    sin_alpha = []
    cos_beta = []
    theta = []
    ang_velo = []
    for n in range(len(List_coor_target)):
      disp.append(math.sqrt((List_coor_target[n] **2)+ (List_coor_target_y[n] **2)))
      if(n< len(List_coor_target) - 1):
        velo_x.append((List_coor_target[n+1] - List_coor_target[n]))
        velo_y.append((List_coor_target_y[n+1] - List_coor_target_y[n]))
        if((List_coor_target_t[n+1] - List_coor_target_t[n]) == 0):
          velocity_x.append(velocity_x[n-1])
          velocity_y.append(velocity_y[n-1])
        else:
          velocity_x.append((List_coor_target[n+1] - List_coor_target[n])/(List_coor_target_t[n+1] - List_coor_target_t[n]))
          velocity_y.append((List_coor_target_y[n+1] - List_coor_target_y[n])/(List_coor_target_t[n+1] - List_coor_target_t[n]))
        abs_velo.append(math.sqrt((velocity_x[n] ** 2) + (velocity_y[n] ** 2)))
        if(abs_velo[n] == 0 and n != 0):
          cos_alpha.append(cos_alpha[n-1])
          sin_alpha.append(sin_alpha[n-1])
          cos_beta.append(cos_beta[n-1])
        elif ( n == 0 and abs_velo[n] == 0):
          '''print("TRUE*********************************************************************")
          velox = (List_coor_target[n+3] - List_coor_target[n+1])
          vx = ((List_coor_target[n+3] - List_coor_target[n+1])/(List_coor_target_t[n+3] - List_coor_target_t[n+1]))
          vy = ((List_coor_target_y[n+3] - List_coor_target_y[n+1])/(List_coor_target_t[n+3] - List_coor_target_t[n+1]))
          absxy = math.sqrt((vx ** 2) + (vy ** 2))
          print(vx)
          print(vy)
          print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", absxy)
          cos_alpha.append((List_coor_target[n+3] - List_coor_target[n+1])/absxy)
          sin_alpha.append((List_coor_target_y[n+3] - List_coor_target_y[n+1])/absxy)
          cos_beta.append(velox / absxy)'''
          cos_alpha.append(np.NaN)
          sin_alpha.append(np.NaN)
          cos_beta.append(np.NaN)
        else:
          cos_alpha.append((List_coor_target[n+1] - List_coor_target[n])/abs_velo[n])
          sin_alpha.append((List_coor_target_y[n+1] - List_coor_target_y[n])/abs_velo[n])
          cos_beta.append(velo_x[n] / abs_velo[n])
        if((List_coor_target[n+1] - List_coor_target[n]) != 0):
          theta.append((math.atan((List_coor_target_y[n+1] - List_coor_target_y[n])/(List_coor_target[n+1] - List_coor_target[n]))))
        if((List_coor_target[n+1] - List_coor_target[n]) == 0):
          theta.append(3.145/2)
    for n in range(len(List_coor_target) - 2):
      ang_velo.append((theta[n+1] - theta[n]))
      if(n < len(List_coor_target) - 3):
        if((List_coor_target_t[n+1] - List_coor_target_t[n]) == 0):
          ac_x.append(ac_x[n-1])
          ac_y.append(ac_y[n-1])
        else:
          ac_x.append((velo_x[n+1] - velo_x[n])/(List_coor_target_t[n+1] - List_coor_target_t[n]))
          ac_y.append((velo_y[n+1] - velo_y[n])/(List_coor_target_t[n+1] - List_coor_target_t[n]))
        abs_ac.append((math.sqrt((ac_x[n] ** 2) + (ac_y[n] ** 2))))
        if(abs_velo[n] == 0 and n != 0):
          cent_ac.append(cent_ac[n-1])
        elif(abs_velo[n] == 0 and n ==0):
          '''velox = (List_coor_target[n+3] - List_coor_target[n+2])
          veloy = (List_coor_target_y[n+3] - List_coor_target_y[n+2])
          vx = ((List_coor_target[n+3] - List_coor_target[n+2])/(List_coor_target_t[n+3] - List_coor_target_t[n+2]))
          vy = ((List_coor_target_y[n+3] - List_coor_target_y[n+2])/(List_coor_target_t[n+3] - List_coor_target_t[n+2]))
          absxy = math.sqrt((vx ** 2) + (vy ** 2))
          if(absxy == 0):
            cent_ac.append(np.NaN)
          else:
            cent_ac.append(((velox * ac_y[n]) - (veloy * ac_x[n]))/absxy)'''
          cent_ac.append(np.NaN)  
        else:
          cent_ac.append(((velo_x[n] * ac_y[n]) - (velo_y[n] * ac_x[n]))/abs_velo[n])
    List_coor_target = List_coor_target.tolist()
    List_coor_target_y = List_coor_target_y.tolist()
    List_coor_target_p = List_coor_target_p.tolist()
    feature_set = disp + velocity_x + velocity_y + abs_velo + cos_alpha + sin_alpha + cos_beta + theta + ang_velo + ac_x + ac_y + abs_ac + cent_ac + List_coor_target + List_coor_target_y +List_coor_target_p
    print(len(feature_set))
    data_set.append(feature_set)
dataframe = pd.DataFrame(data_set)
y = pd.Series(y , index = ind)

##################################### REMOVING NANS #############################
dataframe = dataframe.interpolate(method = 'linear', axis = 1 , limit_direction = 'both')
'''dataframe.interpolate(method = 'linear', axis = 0 , limit_direction = 'backward')
dataframe.fillna(value = 0 , axis = 1)
dataframe.dropna(axis = 1)'''

dataframe.isnull().values.any()

######################################### X TRAIN Y TRAIN DEFINITION #####################################
x_train = dataframe
y_train = y
print(y)

############################################ TEST LOGICAL FEATURE EXTRACTION #################################
data_set = []
file1 = open(r"/content/abcabc/0000/TRAIN/0000f00.txt", 'r')
ind = []
List_coor = []
y = []
no_of_img = 0
datapath = r"/content/abcabc/0055/TEST"
data = os.listdir(datapath)
for dataset in data:
    print(dataset)
    List_coor_target = []
    List_coor_target_y = []
    List_coor_target_azi = []
    List_coor_target_alt = []
    List_coor_target_p = []
    List_coor_target_t = []
    time = 0
    count = 0
    k = 0
    file1.seek(0)
    path = datapath+"/" +dataset
    no_of_img = no_of_img + 1
    name = dataset
    '''for u in range(len(dataset)):
      if(name[u] == "S"):
        temp = u+1
    number = name[temp:len(dataset)-4]
    number = int(number)'''
    if(name[4] == 'v'):
      print(name[4])
      y.append("False")
      print("False")
    else:
      print(name[4])
      y.append("True")
      print("True")
    ind.append(no_of_img)
    file1 = open(path, 'r')
    while True:
        d = 0
        add = 0
        n=0
        line = file1.readline()  
        if not line: 
            break
        if(count == 1):
            k = 1;
        count = count + 1
        if(k == 1):
          for i in range(len(line)):
              if(line[i] == " "):
                  lead = 0
                  lead1 = 0
                  lead2 = 0
                  lead3 = 0
                  lead4 = 0
                  lead5 = 0
                  d=d+1
              if(d == 1 and lead == 0):
                  temp = i+1
                  lead = 1
              if(d == 2 and lead1 == 0):
                  temp2 = i+1
                  lead1 = 1
              if(d == 4 and lead2 == 0):
                  temp3 = i+1
                  lead2 = 1
              '''if(d == 5 and lead3 == 0):
                  temp4 = i+1
                  lead3 = 1
              if(d == 6 and lead4 == 0):
                  temp5 = i+1
                  lead4 = 1'''
              if(d == 3 and lead5 == 0):
                  temp6 = i+1
                  lead5 = 1
          time = time+1
          code = line[0:temp]
          code_y = line[temp:temp2]
          code_p = line[temp2:temp6]
          code_azi = line[temp6:temp3]
          code_alt = line[temp3:len(line)]
          '''code_p = line[temp5:len(line)]'''
          code = int(code)
          code_y = int(code_y)
          code_p = int(code_p)
          code_azi = int(code_azi)
          code_alt = int(code_alt)
          '''code_p = int(code_p)'''
          List_coor.append([no_of_img , time , code , code_y , code_azi , code_alt , code_p])
          List_coor_target.append(code)
          List_coor_target_y.append(code_y)
          List_coor_target_azi.append(code_azi)
          List_coor_target_alt.append(code_alt)
          List_coor_target_p.append(code_p)
          length_target = len(List_coor_target)

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_x = List_coor_target
    f = interpolate.interp1d(x, y_x)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_x = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target = ynew_x

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_y = List_coor_target_y
    f = interpolate.interp1d(x, y_y)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_y = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_y = ynew_y

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_p = List_coor_target_p
    f = interpolate.interp1d(x, y_p)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_p = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_p = ynew_p

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_alt = List_coor_target_alt
    f = interpolate.interp1d(x, y_alt)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_alt = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_alt = ynew_alt

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_azi = List_coor_target_azi
    f = interpolate.interp1d(x, y_azi)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_azi = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_azi = ynew_azi

    for er in range(len(List_coor_target)):
      List_coor_target_t.append(er+1)
    
    '''x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_t = List_coor_target_t
    f = interpolate.interp1d(x, y_t)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_t = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_t = ynew_t'''
    

    disp = []
    velo_x = []
    velo_y = []
    velocity_x = []
    velocity_y = []
    abs_velo = []
    abs_velo = []
    ac_x = []
    ac_y = []
    abs_ac = []
    cent_ac = []
    cos_alpha = []
    sin_alpha = []
    cos_beta = []
    theta = []
    ang_velo = []
    for n in range(len(List_coor_target)):
      disp.append(math.sqrt((List_coor_target[n] **2)+ (List_coor_target_y[n] **2)))
      if(n< len(List_coor_target) - 1):
        velo_x.append((List_coor_target[n+1] - List_coor_target[n]))
        velo_y.append((List_coor_target_y[n+1] - List_coor_target_y[n]))
        if((List_coor_target_t[n+1] - List_coor_target_t[n]) == 0):
          velocity_x.append(velocity_x[n-1])
          velocity_y.append(velocity_y[n-1])
        else:
          velocity_x.append((List_coor_target[n+1] - List_coor_target[n])/(List_coor_target_t[n+1] - List_coor_target_t[n]))
          velocity_y.append((List_coor_target_y[n+1] - List_coor_target_y[n])/(List_coor_target_t[n+1] - List_coor_target_t[n]))
        abs_velo.append(math.sqrt((velocity_x[n] ** 2) + (velocity_y[n] ** 2)))
        if(abs_velo[n] == 0 and n != 0):
          cos_alpha.append(cos_alpha[n-1])
          sin_alpha.append(sin_alpha[n-1])
          cos_beta.append(cos_beta[n-1])
        elif ( n == 0 and abs_velo[n] == 0):
          '''print("TRUE*********************************************************************")
          velox = (List_coor_target[n+3] - List_coor_target[n+1])
          vx = ((List_coor_target[n+3] - List_coor_target[n+1])/(List_coor_target_t[n+3] - List_coor_target_t[n+1]))
          vy = ((List_coor_target_y[n+3] - List_coor_target_y[n+1])/(List_coor_target_t[n+3] - List_coor_target_t[n+1]))
          absxy = math.sqrt((vx ** 2) + (vy ** 2))
          print(vx)
          print(vy)
          print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", absxy)'''
          '''cos_alpha.append((List_coor_target[n+3] - List_coor_target[n+1])/absxy)
          sin_alpha.append((List_coor_target_y[n+3] - List_coor_target_y[n+1])/absxy)
          cos_beta.append(velox / absxy)'''
          cos_alpha.append(np.NaN)
          sin_alpha.append(np.NaN)
          cos_beta.append(np.NaN)
        else:
          cos_alpha.append((List_coor_target[n+1] - List_coor_target[n])/abs_velo[n])
          sin_alpha.append((List_coor_target_y[n+1] - List_coor_target_y[n])/abs_velo[n])
          cos_beta.append(velo_x[n] / abs_velo[n])
        if((List_coor_target[n+1] - List_coor_target[n]) != 0):
          theta.append((math.atan((List_coor_target_y[n+1] - List_coor_target_y[n])/(List_coor_target[n+1] - List_coor_target[n]))))
        if((List_coor_target[n+1] - List_coor_target[n]) == 0):
          theta.append(3.145/2)
    for n in range(len(List_coor_target) - 2):
      ang_velo.append((theta[n+1] - theta[n]))
      if(n < len(List_coor_target) - 3):
        if((List_coor_target_t[n+1] - List_coor_target_t[n]) == 0):
          ac_x.append(ac_x[n-1])
          ac_y.append(ac_y[n-1])
        else:
          ac_x.append((velo_x[n+1] - velo_x[n])/(List_coor_target_t[n+1] - List_coor_target_t[n]))
          ac_y.append((velo_y[n+1] - velo_y[n])/(List_coor_target_t[n+1] - List_coor_target_t[n]))
        abs_ac.append((math.sqrt((ac_x[n] ** 2) + (ac_y[n] ** 2))))
        if(abs_velo[n] == 0 and n != 0):
          cent_ac.append(cent_ac[n-1])
        elif(abs_velo[n] == 0 and n ==0):
          '''velox = (List_coor_target[n+3] - List_coor_target[n+2])
          veloy = (List_coor_target_y[n+3] - List_coor_target_y[n+2])
          vx = ((List_coor_target[n+3] - List_coor_target[n+2])/(List_coor_target_t[n+3] - List_coor_target_t[n+2]))
          vy = ((List_coor_target_y[n+3] - List_coor_target_y[n+2])/(List_coor_target_t[n+3] - List_coor_target_t[n+2]))
          absxy = math.sqrt((vx ** 2) + (vy ** 2))
          cent_ac.append(((velox * ac_y[n]) - (veloy * ac_x[n]))/absxy)'''
          cent_ac.append(np.NaN)
        else:
          cent_ac.append(((velo_x[n] * ac_y[n]) - (velo_y[n] * ac_x[n]))/abs_velo[n])
    List_coor_target = List_coor_target.tolist()
    List_coor_target_y = List_coor_target_y.tolist()
    List_coor_target_p = List_coor_target_p.tolist()
    feature_set = disp + velocity_x + velocity_y + abs_velo + cos_alpha + sin_alpha + cos_beta + theta + ang_velo + ac_x + ac_y + abs_ac + cent_ac + List_coor_target + List_coor_target_y +List_coor_target_p
    print(len(feature_set))
    data_set.append(feature_set)
dataframe = pd.DataFrame(data_set)
y = pd.Series(y , index = ind)

##################################### REMOVING NANS #############################
dataframe = dataframe.interpolate(method = 'linear', axis = 1 , limit_direction = 'both')

############################################ X TEST Y TEST DEFINITION ####################################
x_test = dataframe
y_test = y



from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
#print(y_train)
#print(y_test)
clf=SVC(kernel='linear',probability=True)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
b1=b
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=GaussianNB()
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b2=[]
b2=b

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=BernoulliNB()
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b3=b

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=DecisionTreeClassifier(random_state=0)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b4=b

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=AdaBoostClassifier(n_estimators=100,random_state=0)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b5=b

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=GradientBoostingClassifier(random_state=0)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b6=b


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
clf=RandomForestClassifier(max_depth=9,random_state=0)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b7=b
r=[]
for i in range(len(b)):
  r.append([0,0])

for i in range(len(b)):
  for j in range(2):
    r[i][j]=(b1[i][j]+b2[i][j]+b3[i][j]+b4[i][j]+b5[i][j]+b6[i][j]+b7[i][j])/7
print(r)

r1=[]
for i in range(len(b)):
  if (r[i][0]>r[i][1]):
    r1.append(1)
  else:
    r1.append(0)
print(r1)

###########################3FEATURE EXTRACTION TRAINING SET###########################################333
# MAXIMUM CALCULATION
maximum = 0
file1 = open(r"/content/abcabc/0000/TRAIN/0000f00.txt", 'r')
datapath = r"/content/abcabc/0055"
data = os.listdir(datapath)
for ab in data:
  p1 = datapath + "/" + ab
  print(p1)
  datad = os.listdir(p1)
  for dataset in datad:
      file1.seek(0)
      path = p1+"/" +dataset
      List_coor_target = []
      k=0
      count = 0
      print(path)
      file1 = open(path, 'r')
      length = 0
      while True:
          d = 0
          add = 0
          n=0
          line = file1.readline()  
          if not line: 
              break
          if(count == 1):
              k = 1;
          count = count + 1
          if(k == 1):
              length = length + 1
              for i in range(len(line)):
                  if(line[i] == " "):
                      d=d+1
                  if(d == 0):
                      temp = i+1
                  if(d==1):
                      temp2 = i+1
              code = line[temp:temp2]
              '''print(code)'''
              code = int(code)
              List_coor_target.append(code)
      if(length >= maximum):
          maximum = length
      print(dataset)
      print(length)
print("MAXIMUM" , maximum)
length_query = maximum

###################################### FEATURE EXTRACTION TRAINING SET--------LOGICAL FEATURES#######################
data_set = []
file1 = open(r"/content/abcabc/0000/TRAIN/0000f00.txt", 'r')
ind = []
List_coor = []
y = []
no_of_img = 0
datapath = r"/content/abcabc/0055/TRAIN"
data = os.listdir(datapath)
for dataset in data:
    print(dataset)
    List_coor_target = []
    List_coor_target_y = []
    List_coor_target_azi = []
    List_coor_target_alt = []
    List_coor_target_p = []
    List_coor_target_t = []
    time = 0
    count = 0
    k = 0
    file1.seek(0)
    path = datapath+"/" +dataset
    no_of_img = no_of_img + 1
    name = dataset
    '''for u in range(len(dataset)):
      if(name[u] == "S"):
        temp = u+1
    number = name[temp:len(dataset)-4]
    number = int(number)'''
    if(name[4] == 'v'):
      print(name[4])
      y.append("False")
      print("False")
    else:
      print(name[4])
      y.append("True")
      print("True")
    ind.append(no_of_img)
    file1 = open(path, 'r')
    while True:
        d = 0
        add = 0
        n=0
        line = file1.readline()  
        if not line: 
            break
        if(count == 1):
            k = 1;
        count = count + 1
        if(k == 1):
          for i in range(len(line)):
              if(line[i] == " "):
                  lead = 0
                  lead1 = 0
                  lead2 = 0
                  lead3 = 0
                  lead4 = 0
                  lead5 = 0
                  d=d+1
              if(d == 1 and lead == 0):
                  temp = i+1
                  lead = 1
              if(d == 2 and lead1 == 0):
                  temp2 = i+1
                  lead1 = 1
              if(d == 4 and lead2 == 0):
                  temp3 = i+1
                  lead2 = 1
              '''if(d == 5 and lead3 == 0):
                  temp4 = i+1
                  lead3 = 1
              if(d == 6 and lead4 == 0):
                  temp5 = i+1
                  lead4 = 1'''
              if(d == 3 and lead5 == 0):
                  temp6 = i+1
                  lead5 = 1
          time = time+1
          code = line[0:temp]
          code_y = line[temp:temp2]
          code_p = line[temp2:temp6]
          code_azi = line[temp6:temp3]
          code_alt = line[temp3:len(line)]
          '''code_p = line[temp5:len(line)]'''
          code = int(code)
          code_y = int(code_y)
          code_p = int(code_p)
          code_azi = int(code_azi)
          code_alt = int(code_alt)
          '''code_p = int(code_p)'''
          List_coor.append([no_of_img , time , code , code_y , code_azi , code_alt , code_p])
          List_coor_target.append(code)
          List_coor_target_y.append(code_y)
          List_coor_target_azi.append(code_azi)
          List_coor_target_alt.append(code_alt)
          List_coor_target_p.append(code_p)
          length_target = len(List_coor_target)

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_x = List_coor_target
    f = interpolate.interp1d(x, y_x)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_x = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target = ynew_x

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_y = List_coor_target_y
    f = interpolate.interp1d(x, y_y)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_y = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_y = ynew_y

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_p = List_coor_target_p
    f = interpolate.interp1d(x, y_p)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_p = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_p = ynew_p

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_alt = List_coor_target_alt
    f = interpolate.interp1d(x, y_alt)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_alt = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_alt = ynew_alt

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_azi = List_coor_target_azi
    f = interpolate.interp1d(x, y_azi)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_azi = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_azi = ynew_azi
    
    '''x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_t = List_coor_target_t
    f = interpolate.interp1d(x, y_t)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_t = f(x_new)   # use interpolation function returned by `interp1d`'''
    '''result_x = np.correlate(List_coor_target, List_coor_target, mode='full')
    result_x = result_x[floor(result_x.size/2):].tolist()
    result_y = np.correlate(List_coor_target_y, List_coor_target_y, mode='full')
    result_y = result_y[floor(result_y.size/2):].tolist()
    result_p = np.correlate(List_coor_target_p, List_coor_target_p, mode='full')
    result_p = result_p[floor(result_p.size/2):].tolist()
    result_alt = np.correlate(List_coor_target_alt, List_coor_target_alt, mode='full')
    result_alt = result_alt[floor(result_alt.size/2):].tolist()
    result_azi = np.correlate(List_coor_target_azi, List_coor_target_azi, mode='full')
    result_azi = result_azi[floor(result_azi.size/2):].tolist()

    wavelet_x = dwt(List_coor_target, 'db1')
    wavelet_x = list(wavelet_x)
    wavelet1 = wavelet_x[0].tolist()
    wavelet2 = wavelet_x[1].tolist()
    wavelet_x = wavelet1+wavelet2
    print(len(wavelet_x))
    wavelet_y = dwt(List_coor_target_y, 'db1')
    wavelet_y = list(wavelet_y)
    wavelet1 = wavelet_y[0].tolist()
    wavelet2 = wavelet_y[1].tolist()
    wavelet_y = wavelet1+wavelet2
    print(len(wavelet_y))
    wavelet_p = dwt(List_coor_target_p, 'db1')
    wavelet_p = list(wavelet_p)
    wavelet1 = wavelet_p[0].tolist()
    wavelet2 = wavelet_p[1].tolist()
    wavelet_p = wavelet1+wavelet2
    print(len(wavelet_p))
    wavelet_alt = dwt(List_coor_target_alt, 'db1')
    wavelet_alt = list(wavelet_alt)
    wavelet1 = wavelet_alt[0].tolist()
    wavelet2 = wavelet_alt[1].tolist()
    wavelet_alt = wavelet1+wavelet2
    print(len(wavelet_alt))
    wavelet_azi = dwt(List_coor_target_azi, 'db1')
    wavelet_azi = list(wavelet_azi)
    wavelet1 = wavelet_azi[0].tolist()
    wavelet2 = wavelet_azi[1].tolist()
    wavelet_azi = wavelet1+wavelet2
    print(len(wavelet_azi))'''

    '''List_coor_target_t = ynew_t'''
    dct_x = dct(List_coor_target)
    dct_y = dct(List_coor_target_y)
    dct_p = dct(List_coor_target_p)
    dct_alt = dct(List_coor_target_alt)
    dct_azi = dct(List_coor_target_azi)
    '''dst_x = dst(List_coor_target)
    dst_y = dst(List_coor_target_y)
    dst_p = dst(List_coor_target_p)
    dst_alt = dst(List_coor_target_alt)
    dst_azi = dst(List_coor_target_azi)
    dst_x = dst_x.tolist()
    dst_y = dst_y.tolist()
    dst_p = dst_p.tolist()
    dst_alt = dst_alt.tolist()
    dst_azi = dst_azi.tolist()'''
    dct_x = dct_x.tolist()
    dct_y = dct_y.tolist()
    dct_p = dct_p.tolist()
    dct_alt = dct_alt.tolist()
    dct_azi = dct_azi.tolist()
    '''her_poly_x = eval_hermite(20, List_coor_target, out=None).tolist()
    print(len(her_poly_x))
    her_poly_y = eval_hermite(20, List_coor_target_y, out=None).tolist()
    print(len(her_poly_y))
    her_poly_p = eval_hermite(20, List_coor_target_p, out=None).tolist()
    print(len(her_poly_p))
    her_poly_alt = eval_hermite(20, List_coor_target_alt, out=None).tolist()
    print(len(her_poly_alt))
    her_poly_azi = eval_hermite(20, List_coor_target_azi, out=None).tolist()
    print(len(her_poly_azi))'''
    '''hilbert_x = hilbert(List_coor_target)
    print(hilbert_x)
    print(len(hilbert_x))
    hilbert_y = hilbert(List_coor_target_y)
    print(len(hilbert_y))
    hilbert_p = hilbert(List_coor_target_p)
    print(len(hilbert_p))
    hilbert_alt = hilbert(List_coor_target_alt)
    print(len(hilbert_alt))
    hilbert_azi = hilbert(List_coor_target_azi)
    print(len(hilbert_azi))'''
    feature_set = dct_x + dct_y + dct_p + dct_alt + dct_azi
    print(len(feature_set))
    data_set.append(feature_set)
dataframe = pd.DataFrame(data_set)
y = pd.Series(y , index = ind)

dataframe = dataframe.interpolate(method = 'linear', axis = 1 , limit_direction = 'both')
dataframe

dataframe.isnull().values.any()

######################################### X TRAIN Y TRAIN DEFINITION #####################################
x_train = dataframe
y_train = y

############################################ TEST LOGICAL FEATURE EXTRACTION #################################
data_set = []
file1 = open(r"/content/abcabc/0000/TRAIN/0000f00.txt", 'r')
ind = []
List_coor = []
y = []
no_of_img = 0
datapath = r"/content/abcabc/0055/TEST"
data = os.listdir(datapath)
for dataset in data:
    print(dataset)
    List_coor_target = []
    List_coor_target_y = []
    List_coor_target_azi = []
    List_coor_target_alt = []
    List_coor_target_p = []
    List_coor_target_t = []
    time = 0
    count = 0
    k = 0
    file1.seek(0)
    path = datapath+"/" +dataset
    no_of_img = no_of_img + 1
    name = dataset
    '''for u in range(len(dataset)):
      if(name[u] == "S"):
        temp = u+1
    number = name[temp:len(dataset)-4]
    number = int(number)'''
    if(name[4] == 'v'):
      print(name[4])
      y.append("False")
      print("False")
    else:
      print(name[4])
      y.append("True")
      print("True")
    ind.append(no_of_img)
    file1 = open(path, 'r')
    while True:
        d = 0
        add = 0
        n=0
        line = file1.readline()  
        if not line: 
            break
        if(count == 1):
            k = 1;
        count = count + 1
        if(k == 1):
          for i in range(len(line)):
              if(line[i] == " "):
                  lead = 0
                  lead1 = 0
                  lead2 = 0
                  lead3 = 0
                  lead4 = 0
                  lead5 = 0
                  d=d+1
              if(d == 1 and lead == 0):
                  temp = i+1
                  lead = 1
              if(d == 2 and lead1 == 0):
                  temp2 = i+1
                  lead1 = 1
              if(d == 4 and lead2 == 0):
                  temp3 = i+1
                  lead2 = 1
              '''if(d == 5 and lead3 == 0):
                  temp4 = i+1
                  lead3 = 1
              if(d == 6 and lead4 == 0):
                  temp5 = i+1
                  lead4 = 1'''
              if(d == 3 and lead5 == 0):
                  temp6 = i+1
                  lead5 = 1
          time = time+1
          code = line[0:temp]
          code_y = line[temp:temp2]
          code_p = line[temp2:temp6]
          code_azi = line[temp6:temp3]
          code_alt = line[temp3:len(line)]
          '''code_p = line[temp5:len(line)]'''
          code = int(code)
          code_y = int(code_y)
          code_p = int(code_p)
          code_azi = int(code_azi)
          code_alt = int(code_alt)
          '''code_p = int(code_p)'''
          List_coor.append([no_of_img , time , code , code_y , code_azi , code_alt , code_p])
          List_coor_target.append(code)
          List_coor_target_y.append(code_y)
          List_coor_target_azi.append(code_azi)
          List_coor_target_alt.append(code_alt)
          List_coor_target_p.append(code_p)
          length_target = len(List_coor_target)

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_x = List_coor_target
    f = interpolate.interp1d(x, y_x)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_x = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target = ynew_x

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_y = List_coor_target_y
    f = interpolate.interp1d(x, y_y)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_y = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_y = ynew_y

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_p = List_coor_target_p
    f = interpolate.interp1d(x, y_p)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_p = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_p = ynew_p

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_alt = List_coor_target_alt
    f = interpolate.interp1d(x, y_alt)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_alt = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_alt = ynew_alt

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_azi = List_coor_target_azi
    f = interpolate.interp1d(x, y_azi)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_azi = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_azi = ynew_azi
    
    '''x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_t = List_coor_target_t
    f = interpolate.interp1d(x, y_t)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_t = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_t = ynew_t'''
    '''result_x = np.correlate(List_coor_target, List_coor_target, mode='full')
    result_x = result_x[floor(result_x.size/2):].tolist()
    result_y = np.correlate(List_coor_target_y, List_coor_target_y, mode='full')
    result_y = result_y[floor(result_y.size/2):].tolist()
    result_p = np.correlate(List_coor_target_p, List_coor_target_p, mode='full')
    result_p = result_p[floor(result_p.size/2):].tolist()
    result_alt = np.correlate(List_coor_target_alt, List_coor_target_alt, mode='full')
    result_alt = result_alt[floor(result_alt.size/2):].tolist()
    result_azi = np.correlate(List_coor_target_azi, List_coor_target_azi, mode='full')
    result_azi = result_azi[floor(result_azi.size/2):].tolist()
    wavelet_x = dwt(List_coor_target, 'db1')
    wavelet_x = list(wavelet_x)
    wavelet1 = wavelet_x[0].tolist()
    wavelet2 = wavelet_x[1].tolist()
    wavelet_x = wavelet1+wavelet2
    print(len(wavelet_x))
    wavelet_y = dwt(List_coor_target_y, 'db1')
    wavelet_y = list(wavelet_y)
    wavelet1 = wavelet_y[0].tolist()
    wavelet2 = wavelet_y[1].tolist()
    wavelet_y = wavelet1+wavelet2
    print(len(wavelet_y))
    wavelet_p = dwt(List_coor_target_p, 'db1')
    wavelet_p = list(wavelet_p)
    wavelet1 = wavelet_p[0].tolist()
    wavelet2 = wavelet_p[1].tolist()
    wavelet_p = wavelet1+wavelet2
    print(len(wavelet_p))
    wavelet_alt = dwt(List_coor_target_alt, 'db1')
    wavelet_alt = list(wavelet_alt)
    wavelet1 = wavelet_alt[0].tolist()
    wavelet2 = wavelet_alt[1].tolist()
    wavelet_alt = wavelet1+wavelet2
    print(len(wavelet_alt))
    wavelet_azi = dwt(List_coor_target_azi, 'db1')
    wavelet_azi = list(wavelet_azi)
    wavelet1 = wavelet_azi[0].tolist()
    wavelet2 = wavelet_azi[1].tolist()
    wavelet_azi = wavelet1+wavelet2
    print(len(wavelet_azi))'''
    dct_x = dct(List_coor_target)
    dct_y = dct(List_coor_target_y)
    dct_p = dct(List_coor_target_p)
    dct_alt = dct(List_coor_target_alt)
    dct_azi = dct(List_coor_target_azi)
    '''dst_x = dst(List_coor_target)
    dst_y = dst(List_coor_target_y)
    dst_p = dst(List_coor_target_p)
    dst_alt = dst(List_coor_target_alt)
    dst_azi = dst(List_coor_target_azi)
    dst_x = dst_x.tolist()
    dst_y = dst_y.tolist()
    dst_p = dst_p.tolist()
    dst_alt = dst_alt.tolist()
    dst_azi = dst_azi.tolist()'''
    dct_x = dct_x.tolist()
    dct_y = dct_y.tolist()
    dct_p = dct_p.tolist()
    dct_alt = dct_alt.tolist()
    dct_azi = dct_azi.tolist()
    '''her_poly_x = eval_hermite(20, List_coor_target, out=None).tolist()
    print(len(her_poly_x))
    her_poly_y = eval_hermite(20, List_coor_target_y, out=None).tolist()
    print(len(her_poly_y))
    her_poly_p = eval_hermite(20, List_coor_target_p, out=None).tolist()
    print(len(her_poly_p))
    her_poly_alt = eval_hermite(20, List_coor_target_alt, out=None).tolist()
    print(len(her_poly_alt))
    her_poly_azi = eval_hermite(20, List_coor_target_azi, out=None).tolist()
    print(len(her_poly_azi))'''
    feature_set = dct_x + dct_y + dct_p + dct_alt + dct_azi
    print(len(feature_set))
    data_set.append(feature_set)
dataframe = pd.DataFrame(data_set)
y = pd.Series(y , index = ind)

dataframe = dataframe.interpolate(method = 'linear', axis = 1 , limit_direction = 'both')
dataframe

############################################ X TEST Y TEST DEFINITION ####################################
x_test = dataframe
y_test = y
    
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
#print(y_train)
#print(y_test)
clf=SVC(kernel='linear',probability=True)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
b1=b
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=GaussianNB()
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b2=[]
b2=b

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=BernoulliNB()
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b3=b

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=DecisionTreeClassifier(random_state=0)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b4=b

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=AdaBoostClassifier(n_estimators=100,random_state=0)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b5=b

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=GradientBoostingClassifier(random_state=0)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b6=b


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
clf=RandomForestClassifier(max_depth=9,random_state=0)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b7=b
r=[]
for i in range(len(b)):
  r.append([0,0])

for i in range(len(b)):
  for j in range(2):
    r[i][j]=(b1[i][j]+b2[i][j]+b3[i][j]+b4[i][j]+b5[i][j]+b6[i][j]+b7[i][j])/7
print(r)

r2=[]
for i in range(len(b)):
  if (r[i][0]>r[i][1]):
    r2.append(1)
  else:
    r2.append(0)
print(r2)

###########################3FEATURE EXTRACTION TRAINING SET###########################################333
# MAXIMUM CALCULATION
maximum = 0
file1 = open(r"/content/abcabc/0000/TRAIN/0000f00.txt", 'r')
datapath = r"/content/abcabc/0055"
data = os.listdir(datapath)
for ab in data:
  p1 = datapath + "/" + ab
  print(p1)
  datad = os.listdir(p1)
  for dataset in datad:
      file1.seek(0)
      path = p1+"/" +dataset
      List_coor_target = []
      k=0
      count = 0
      print(path)
      file1 = open(path, 'r')
      length = 0
      while True:
          d = 0
          add = 0
          n=0
          line = file1.readline()  
          if not line: 
              break
          if(count == 1):
              k = 1;
          count = count + 1
          if(k == 1):
              length = length + 1
              for i in range(len(line)):
                  if(line[i] == " "):
                      d=d+1
                  if(d == 0):
                      temp = i+1
                  if(d==1):
                      temp2 = i+1
              code = line[temp:temp2]
              '''print(code)'''
              code = int(code)
              List_coor_target.append(code)
      if(length >= maximum):
          maximum = length
      print(dataset)
      print(length)
print("MAXIMUM" , maximum)
length_query = maximum

###################################### FEATURE EXTRACTION TRAINING SET--------LOGICAL FEATURES#######################
data_set = []
file1 = open(r"/content/abcabc/0000/TRAIN/0000f00.txt", 'r')
ind = []
List_coor = []
y = []
no_of_img = 0
datapath = r"/content/abcabc/0055/TRAIN"
data = os.listdir(datapath)
for dataset in data:
    print(dataset)
    List_coor_target = []
    List_coor_target_y = []
    List_coor_target_azi = []
    List_coor_target_alt = []
    List_coor_target_p = []
    List_coor_target_t = []
    time = 0
    count = 0
    k = 0
    file1.seek(0)
    path = datapath+"/" +dataset
    no_of_img = no_of_img + 1
    name = dataset
    '''for u in range(len(dataset)):
      if(name[u] == "S"):
        temp = u+1
    number = name[temp:len(dataset)-4]
    number = int(number)'''
    if(name[4] == 'v'):
      print(name[4])
      y.append("False")
      print("False")
    else:
      print(name[4])
      y.append("True")
      print("True")
    ind.append(no_of_img)
    file1 = open(path, 'r')
    while True:
        d = 0
        add = 0
        n=0
        line = file1.readline()  
        if not line: 
            break
        if(count == 1):
            k = 1;
        count = count + 1
        if(k == 1):
          for i in range(len(line)):
              if(line[i] == " "):
                  lead = 0
                  lead1 = 0
                  lead2 = 0
                  lead3 = 0
                  lead4 = 0
                  lead5 = 0
                  d=d+1
              if(d == 1 and lead == 0):
                  temp = i+1
                  lead = 1
              if(d == 2 and lead1 == 0):
                  temp2 = i+1
                  lead1 = 1
              if(d == 4 and lead2 == 0):
                  temp3 = i+1
                  lead2 = 1
              '''if(d == 5 and lead3 == 0):
                  temp4 = i+1
                  lead3 = 1
              if(d == 6 and lead4 == 0):
                  temp5 = i+1
                  lead4 = 1'''
              if(d == 3 and lead5 == 0):
                  temp6 = i+1
                  lead5 = 1
          time = time+1
          code = line[0:temp]
          code_y = line[temp:temp2]
          code_p = line[temp2:temp6]
          code_azi = line[temp6:temp3]
          code_alt = line[temp3:len(line)]
          '''code_p = line[temp5:len(line)]'''
          code = int(code)
          code_y = int(code_y)
          code_p = int(code_p)
          code_azi = int(code_azi)
          code_alt = int(code_alt)
          '''code_p = int(code_p)'''
          List_coor.append([no_of_img , time , code , code_y , code_azi , code_alt , code_p])
          List_coor_target.append(code)
          List_coor_target_y.append(code_y)
          List_coor_target_azi.append(code_azi)
          List_coor_target_alt.append(code_alt)
          List_coor_target_p.append(code_p)
          length_target = len(List_coor_target)

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_x = List_coor_target
    f = interpolate.interp1d(x, y_x)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_x = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target = ynew_x

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_y = List_coor_target_y
    f = interpolate.interp1d(x, y_y)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_y = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_y = ynew_y

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_p = List_coor_target_p
    f = interpolate.interp1d(x, y_p)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_p = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_p = ynew_p

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_alt = List_coor_target_alt
    f = interpolate.interp1d(x, y_alt)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_alt = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_alt = ynew_alt

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_azi = List_coor_target_azi
    f = interpolate.interp1d(x, y_azi)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_azi = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_azi = ynew_azi
    
    '''x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_t = List_coor_target_t
    f = interpolate.interp1d(x, y_t)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_t = f(x_new)   # use interpolation function returned by `interp1d`'''
    result_x = np.correlate(List_coor_target, List_coor_target, mode='full')
    result_x = result_x[floor(result_x.size/2):].tolist()
    result_y = np.correlate(List_coor_target_y, List_coor_target_y, mode='full')
    result_y = result_y[floor(result_y.size/2):].tolist()
    result_p = np.correlate(List_coor_target_p, List_coor_target_p, mode='full')
    result_p = result_p[floor(result_p.size/2):].tolist()
    result_alt = np.correlate(List_coor_target_alt, List_coor_target_alt, mode='full')
    result_alt = result_alt[floor(result_alt.size/2):].tolist()
    result_azi = np.correlate(List_coor_target_azi, List_coor_target_azi, mode='full')
    result_azi = result_azi[floor(result_azi.size/2):].tolist()

    '''wavelet_x = dwt(List_coor_target, 'db1')
    wavelet_x = list(wavelet_x)
    wavelet1 = wavelet_x[0].tolist()
    wavelet2 = wavelet_x[1].tolist()
    wavelet_x = wavelet1+wavelet2
    print(len(wavelet_x))
    wavelet_y = dwt(List_coor_target_y, 'db1')
    wavelet_y = list(wavelet_y)
    wavelet1 = wavelet_y[0].tolist()
    wavelet2 = wavelet_y[1].tolist()
    wavelet_y = wavelet1+wavelet2
    print(len(wavelet_y))
    wavelet_p = dwt(List_coor_target_p, 'db1')
    wavelet_p = list(wavelet_p)
    wavelet1 = wavelet_p[0].tolist()
    wavelet2 = wavelet_p[1].tolist()
    wavelet_p = wavelet1+wavelet2
    print(len(wavelet_p))
    wavelet_alt = dwt(List_coor_target_alt, 'db1')
    wavelet_alt = list(wavelet_alt)
    wavelet1 = wavelet_alt[0].tolist()
    wavelet2 = wavelet_alt[1].tolist()
    wavelet_alt = wavelet1+wavelet2
    print(len(wavelet_alt))
    wavelet_azi = dwt(List_coor_target_azi, 'db1')
    wavelet_azi = list(wavelet_azi)
    wavelet1 = wavelet_azi[0].tolist()
    wavelet2 = wavelet_azi[1].tolist()
    wavelet_azi = wavelet1+wavelet2
    print(len(wavelet_azi))'''

    '''List_coor_target_t = ynew_t
    dct_x = dct(List_coor_target)
    dct_y = dct(List_coor_target_y)
    dct_p = dct(List_coor_target_p)
    dct_alt = dct(List_coor_target_alt)
    dct_azi = dct(List_coor_target_azi)'''
    '''dst_x = dst(List_coor_target)
    dst_y = dst(List_coor_target_y)
    dst_p = dst(List_coor_target_p)
    dst_alt = dst(List_coor_target_alt)
    dst_azi = dst(List_coor_target_azi)
    dst_x = dst_x.tolist()
    dst_y = dst_y.tolist()
    dst_p = dst_p.tolist()
    dst_alt = dst_alt.tolist()
    dst_azi = dst_azi.tolist()'''
    '''dct_x = dct_x.tolist()
    dct_y = dct_y.tolist()
    dct_p = dct_p.tolist()
    dct_alt = dct_alt.tolist()
    dct_azi = dct_azi.tolist()'''
    '''her_poly_x = eval_hermite(20, List_coor_target, out=None).tolist()
    print(len(her_poly_x))
    her_poly_y = eval_hermite(20, List_coor_target_y, out=None).tolist()
    print(len(her_poly_y))
    her_poly_p = eval_hermite(20, List_coor_target_p, out=None).tolist()
    print(len(her_poly_p))
    her_poly_alt = eval_hermite(20, List_coor_target_alt, out=None).tolist()
    print(len(her_poly_alt))
    her_poly_azi = eval_hermite(20, List_coor_target_azi, out=None).tolist()
    print(len(her_poly_azi))'''
    '''hilbert_x = hilbert(List_coor_target)
    print(hilbert_x)
    print(len(hilbert_x))
    hilbert_y = hilbert(List_coor_target_y)
    print(len(hilbert_y))
    hilbert_p = hilbert(List_coor_target_p)
    print(len(hilbert_p))
    hilbert_alt = hilbert(List_coor_target_alt)
    print(len(hilbert_alt))
    hilbert_azi = hilbert(List_coor_target_azi)
    print(len(hilbert_azi))'''
    feature_set = result_x + result_y + result_p + result_alt + result_azi
    print(len(feature_set))
    data_set.append(feature_set)
dataframe = pd.DataFrame(data_set)
y = pd.Series(y , index = ind)


dataframe = dataframe.interpolate(method = 'linear', axis = 1 , limit_direction = 'both')
dataframe
dataframe.isnull().values.any()
######################################### X TRAIN Y TRAIN DEFINITION #####################################
x_train = dataframe
y_train = y
print(y)

############################################ TEST LOGICAL FEATURE EXTRACTION #################################
data_set = []
file1 = open(r"/content/abcabc/0000/TRAIN/0000f00.txt", 'r')
ind = []
List_coor = []
y = []
no_of_img = 0
datapath = r"/content/abcabc/0055/TEST"
data = os.listdir(datapath)
for dataset in data:
    print(dataset)
    List_coor_target = []
    List_coor_target_y = []
    List_coor_target_azi = []
    List_coor_target_alt = []
    List_coor_target_p = []
    List_coor_target_t = []
    time = 0
    count = 0
    k = 0
    file1.seek(0)
    path = datapath+"/" +dataset
    no_of_img = no_of_img + 1
    name = dataset
    '''for u in range(len(dataset)):
      if(name[u] == "S"):
        temp = u+1
    number = name[temp:len(dataset)-4]
    number = int(number)'''
    if(name[4] == 'v'):
      print(name[4])
      y.append("False")
      print("False")
    else:
      print(name[4])
      y.append("True")
      print("True")
    ind.append(no_of_img)
    file1 = open(path, 'r')
    while True:
        d = 0
        add = 0
        n=0
        line = file1.readline()  
        if not line: 
            break
        if(count == 1):
            k = 1;
        count = count + 1
        if(k == 1):
          for i in range(len(line)):
              if(line[i] == " "):
                  lead = 0
                  lead1 = 0
                  lead2 = 0
                  lead3 = 0
                  lead4 = 0
                  lead5 = 0
                  d=d+1
              if(d == 1 and lead == 0):
                  temp = i+1
                  lead = 1
              if(d == 2 and lead1 == 0):
                  temp2 = i+1
                  lead1 = 1
              if(d == 4 and lead2 == 0):
                  temp3 = i+1
                  lead2 = 1
              '''if(d == 5 and lead3 == 0):
                  temp4 = i+1
                  lead3 = 1
              if(d == 6 and lead4 == 0):
                  temp5 = i+1
                  lead4 = 1'''
              if(d == 3 and lead5 == 0):
                  temp6 = i+1
                  lead5 = 1
          time = time+1
          code = line[0:temp]
          code_y = line[temp:temp2]
          code_p = line[temp2:temp6]
          code_azi = line[temp6:temp3]
          code_alt = line[temp3:len(line)]
          '''code_p = line[temp5:len(line)]'''
          code = int(code)
          code_y = int(code_y)
          code_p = int(code_p)
          code_azi = int(code_azi)
          code_alt = int(code_alt)
          '''code_p = int(code_p)'''
          List_coor.append([no_of_img , time , code , code_y , code_azi , code_alt , code_p])
          List_coor_target.append(code)
          List_coor_target_y.append(code_y)
          List_coor_target_azi.append(code_azi)
          List_coor_target_alt.append(code_alt)
          List_coor_target_p.append(code_p)
          length_target = len(List_coor_target)

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_x = List_coor_target
    f = interpolate.interp1d(x, y_x)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_x = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target = ynew_x

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_y = List_coor_target_y
    f = interpolate.interp1d(x, y_y)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_y = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_y = ynew_y

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_p = List_coor_target_p
    f = interpolate.interp1d(x, y_p)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_p = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_p = ynew_p

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_alt = List_coor_target_alt
    f = interpolate.interp1d(x, y_alt)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_alt = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_alt = ynew_alt

    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_azi = List_coor_target_azi
    f = interpolate.interp1d(x, y_azi)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_azi = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_azi = ynew_azi
    
    '''x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_t = List_coor_target_t
    f = interpolate.interp1d(x, y_t)
    size_extra = maximum - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_t = f(x_new)   # use interpolation function returned by `interp1d`
    List_coor_target_t = ynew_t'''
    result_x = np.correlate(List_coor_target, List_coor_target, mode='full')
    result_x = result_x[floor(result_x.size/2):].tolist()
    result_y = np.correlate(List_coor_target_y, List_coor_target_y, mode='full')
    result_y = result_y[floor(result_y.size/2):].tolist()
    result_p = np.correlate(List_coor_target_p, List_coor_target_p, mode='full')
    result_p = result_p[floor(result_p.size/2):].tolist()
    result_alt = np.correlate(List_coor_target_alt, List_coor_target_alt, mode='full')
    result_alt = result_alt[floor(result_alt.size/2):].tolist()
    result_azi = np.correlate(List_coor_target_azi, List_coor_target_azi, mode='full')
    result_azi = result_azi[floor(result_azi.size/2):].tolist()
    '''wavelet_x = dwt(List_coor_target, 'db1')
    wavelet_x = list(wavelet_x)
    wavelet1 = wavelet_x[0].tolist()
    wavelet2 = wavelet_x[1].tolist()
    wavelet_x = wavelet1+wavelet2
    print(len(wavelet_x))
    wavelet_y = dwt(List_coor_target_y, 'db1')
    wavelet_y = list(wavelet_y)
    wavelet1 = wavelet_y[0].tolist()
    wavelet2 = wavelet_y[1].tolist()
    wavelet_y = wavelet1+wavelet2
    print(len(wavelet_y))
    wavelet_p = dwt(List_coor_target_p, 'db1')
    wavelet_p = list(wavelet_p)
    wavelet1 = wavelet_p[0].tolist()
    wavelet2 = wavelet_p[1].tolist()
    wavelet_p = wavelet1+wavelet2
    print(len(wavelet_p))
    wavelet_alt = dwt(List_coor_target_alt, 'db1')
    wavelet_alt = list(wavelet_alt)
    wavelet1 = wavelet_alt[0].tolist()
    wavelet2 = wavelet_alt[1].tolist()
    wavelet_alt = wavelet1+wavelet2
    print(len(wavelet_alt))
    wavelet_azi = dwt(List_coor_target_azi, 'db1')
    wavelet_azi = list(wavelet_azi)
    wavelet1 = wavelet_azi[0].tolist()
    wavelet2 = wavelet_azi[1].tolist()
    wavelet_azi = wavelet1+wavelet2
    print(len(wavelet_azi))'''
    '''dct_x = dct(List_coor_target)
    dct_y = dct(List_coor_target_y)
    dct_p = dct(List_coor_target_p)
    dct_alt = dct(List_coor_target_alt)
    dct_azi = dct(List_coor_target_azi)'''
    '''dst_x = dst(List_coor_target)
    dst_y = dst(List_coor_target_y)
    dst_p = dst(List_coor_target_p)
    dst_alt = dst(List_coor_target_alt)
    dst_azi = dst(List_coor_target_azi)
    dst_x = dst_x.tolist()
    dst_y = dst_y.tolist()
    dst_p = dst_p.tolist()
    dst_alt = dst_alt.tolist()
    dst_azi = dst_azi.tolist()'''
    '''dct_x = dct_x.tolist()
    dct_y = dct_y.tolist()
    dct_p = dct_p.tolist()
    dct_alt = dct_alt.tolist()
    dct_azi = dct_azi.tolist()'''
    '''her_poly_x = eval_hermite(20, List_coor_target, out=None).tolist()
    print(len(her_poly_x))
    her_poly_y = eval_hermite(20, List_coor_target_y, out=None).tolist()
    print(len(her_poly_y))
    her_poly_p = eval_hermite(20, List_coor_target_p, out=None).tolist()
    print(len(her_poly_p))
    her_poly_alt = eval_hermite(20, List_coor_target_alt, out=None).tolist()
    print(len(her_poly_alt))
    her_poly_azi = eval_hermite(20, List_coor_target_azi, out=None).tolist()
    print(len(her_poly_azi))'''
    feature_set = result_x + result_y + result_p + result_alt + result_azi
    print(len(feature_set))
    data_set.append(feature_set)
dataframe = pd.DataFrame(data_set)
y = pd.Series(y , index = ind)

dataframe = dataframe.interpolate(method = 'linear', axis = 1 , limit_direction = 'both')
dataframe

############################################ X TEST Y TEST DEFINITION ####################################
x_test = dataframe
y_test = y
    
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
#print(y_train)
#print(y_test)
clf=SVC(kernel='linear',probability=True)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
b1=b
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=GaussianNB()
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b2=[]
b2=b

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=BernoulliNB()
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b3=b

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=DecisionTreeClassifier(random_state=0)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b4=b

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=AdaBoostClassifier(n_estimators=100,random_state=0)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b5=b

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#print(y_train)
#print(y_test)
clf=GradientBoostingClassifier(random_state=0)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b6=b


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
clf=RandomForestClassifier(max_depth=9,random_state=0)
clf.fit(x_train,y_train)
a=clf.predict(x_test)
b = clf.predict_proba(x_test)
print(y_test)
print(a)
print(b)
b7=b
r=[]
for i in range(len(b)):
  r.append([0,0])

for i in range(len(b)):
  for j in range(2):
    r[i][j]=(b1[i][j]+b2[i][j]+b3[i][j]+b4[i][j]+b5[i][j]+b6[i][j]+b7[i][j])/7
print(r)

r3=[]
for i in range(len(b)):
  if (r[i][0]>r[i][1]):
    r3.append(1)
  else:
    r3.append(0)
print(r3)

print(r1)
print(r2)
print(r3)

result=[]
for i in range(len(r1)):
  t=0
  f=0
  if (r1[i]==1):
    f=f+1
  else:
    t=t+1
  if (r2[i]==1):
    f=f+1
  else:
    t=t+1
  if (r3[i]==1):
    f=f+1
  else:
    t=t+1
  if (t>f):
    result.append(0)
  else:
    result.append(1)
print(result)
