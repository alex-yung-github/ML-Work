import csv
listOfVals = ['v1005', 'v1013', 'v1026', 'v1028', 'v1030', 'v1031', 'v1032', 'v1038', 'v1041', 'v1061', 'v1066', 'v1067', 'v1068', 'v771', 'v783', 'v786', 'v787', 'v796', 'v799', 'v801', 'v805', 'v807', 'v809', 'v818', 'v830', 'v837', 'v843', 'v844', 'v848', 'v849', 'v850', 'v851', 'v853', 'v856', 'v857', 'v867', 'v868', 'v869', 'v870', 'v874', 'v878', 'v880', 'v886', 'v889', 'v890', 'v895', 'v901', 'v902', 'v905', 'v906', 'v909', 'v912', 'v916', 'v917', 'v920', 'v923', 'v928', 'v931', 'v936', 'v937', 'v938', 'v945', 'v954', 'v957', 'v959', 'v960', 'v961', 'v965', 'v972', 'v975', 'v976', 'v977', 'v979', 'v994', 'v996', 'v998']

totalList = []
count = 0
firstrow = ""
rowOfRows = []
with open("housevotes-74-no-missing-values-testing-set (1).csv", "r") as f:
    for line in f:
        if(count == 0):
            count+=1
            firstrow = line
            temp = line.split(",")
            # print(temp[0])
            for i in range(len(temp)):
                val = temp[i].strip()
                if(val in listOfVals):
                    totalList.append(i)
            print(len(temp))
        else:
            newrow = []
            tempRow = line.split(",")
            for i in range(len(tempRow)):
                # if(i)
                print()
            print(len(tempRow))
    

print(firstrow)
print(totalList)
    

# with open('new.csv' ,'w', newline='') as f:
#     csvwriter = csv.writer(f) 
#     csvwriter.writerow(["sepallength","sepalwidth","petallength","petalwidth","class"])
#     for i in range(len(X_test)):
#         temp = y_train[i]
#         toReturn = np.append(temp, y_test[i])
#         csvwriter.writerow(toReturn)


