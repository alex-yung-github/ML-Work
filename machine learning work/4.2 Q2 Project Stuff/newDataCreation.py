import random
import numpy as np
import math
import json
import random
import statistics

# ranges to best skew the dataset as discerned from the graphs of data
# beegest:
# 65-80,18-24,5.1-10,28.3-55.5,28-56

# beeg:
# 120-144,0.5-7,0.2-2.5,28.3-55,27-56

# big:
# 65-75,15-25,5.6-9,28.3-55,27-56

N = 13 #number of data in each range you want to create
MULTIPLIER = 100

#for thyroid dataset
beegestRange = [(650,800), (180,240), (50,100), (280,560),(280,570)]
beegRange = [(1200,1440), (5,70), (2,25), (280,550),(270,560)]
bigRange = [(650,750), (150,250), (56,90), (280,550),(270,560)]
#QUICK NOTE: the names of the ranges and the classes are arbitrary because we made up class names temporarily

#for voice genre dataset
decimaldmalerange=[(.02, .1), (.01, .03), (.01, .07), (.2, .25), (.16, .27),
(.01, .05), (4, 6), (2, 3), (.7, .85), (.01, .24), (0, .28), (.2,.27),
(.15,.25), (.01, .6), (.18, .23), (1.5, 2.5), (.1,.2), (10.1, 11), (9,12),
(.3,.5)]
decimalfemalerange = [(.02, .1), (.04, .1), (.011, .015),
(.06, .15), (.15,.19), (.06,.17),
(.14,.4), (2, 3), (.9, 1), (.3, .8),
(0, .28), (.05, .13), (.02, .15),
(.01, .05), (.24, .28), (.01, 1.5), (0, .12), (0, 5),
(0, 8), (0, .3)]
malerange = [(2 , 10), (1 , 3), (1 , 7), (20 , 25), (16 , 27), (1 , 5), (400 , 600), (200 , 300), (70 , 85), (1 , 24), (0 , 28), (20 , 27), (15 , 25), (1 , 60), (18 , 23), (150 , 250), (10 , 20), (1010 , 1100), (900 , 1200), (30 , 50)]
femalerange = [(2 , 10), (4 , 10), (1 , 1), (6 , 15), (15 , 19), (6 , 17), (14 , 40), (200 , 300), (90 , 100), (30 , 80), (0 , 28), (5 , 13), (2 , 15), (1 , 5), (24 , 28), (1 , 150), (0 , 12), (0 , 500), (0 , 800), (0 , 30)]

def getVals(r, name):
    listOfLists = []
    for vals in r:
        b1 = vals[0]
        b2 = vals[1]
        # randomNums = random.randint(b1, b2)
        randomNums = []
        for l in range(N):
            w = 1/MULTIPLIER
            temp = '%.9f'%(random.randint(b1, b2) * w)
            randomNums.append(float(temp))
        listOfLists.append(randomNums)
    # print(listOfLists)
    allData = []
    for x in range(len(listOfLists[0])):
        datapoint = []
        for h in range(len(listOfLists)):
            temp = listOfLists[h]
            datapoint.append(temp[x])
        data = tuple(datapoint)
        allData.append(data)
    # print(allData)
    for i in allData:
        for x in range(len(i)):
            print(i[x],  end = ', ')
        print(name)

def main():
    getVals(beegestRange, "beegest")
    getVals(beegRange, "beeg")
    getVals(bigRange, "big")
    # getVals(malerange, "male")
    # getVals(femalerange, "female")

def decimalchange(r):
    for i in range(len(r)):
        print("(", end = '')
        print(int(r[i][0] * MULTIPLIER), ",", int(r[i][1] * MULTIPLIER), end = "")
        print(")", end = ", ")

main() #prints datapoints to easily copy and paste into csv file
# decimalchange(decimalfemalerange)

