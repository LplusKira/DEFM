import sys
import time
import random


DATA_PATH_PREFIX = '../data/'
GEN_DATA_PATH_PREFIX = '../gendata/'

### load data ###
def loadRatingFile(filename):
    ratingList = []
    with open(filename, 'r') as f:
        for line in f:
            uid, mid, r, t = line.strip().split('::')
            uid, mid, r, t = int(uid), int(mid), int(r), int(t)
            ratingList.append([uid, mid, r, t])
    return ratingList


def loadUserFile(filename):
    userList = []
    with open(filename, 'r') as f:
        for line in f:
            uid, gender, age, occ, zip_code = line.strip().split('::')
            uid, age, occ = int(uid), int(age), int(occ)
            userList.append([uid, gender, age, occ])
    return userList


### process data ###
def extendColumn(labelList, colIdx, demoList):
    extendLabelList = []
    demoDict = {demo:index for index, demo in enumerate(demoList)}
    for label in labelList:
        extendLabel = label[:]
        extendColumn = [0 for i in range(len(demoList))]
        extendColumn[demoDict[label[colIdx]]] = 1
        extendLabel[colIdx] = extendColumn
        extendLabelList.append(extendLabel)
    return extendLabelList


def mergeColumn(labelList):
    mergeLabelList = []
    columnNum = len(labelList[0])
    for label in labelList:
        mergeLabel = []
        mergeLabel.append(label[0])
        for j in range(1, columnNum):
            mergeLabel += label[j]
        mergeLabelList.append(mergeLabel)
    return mergeLabelList


def collectRatingToUser(ratingList):
    collectRatingList = []
    userDict = {}
    index = 0
    for rating in ratingList:
        uid, mid, r, t = rating
        if uid not in userDict:
            userDict[uid] = index
            collectRatingList.append([uid])
            index += 1
    for rating in ratingList:
        uid, mid, r, t = rating
        collectRatingList[userDict[uid]].append([mid, r, t])
    return collectRatingList


def splitTrainTestData(idList, labelList, ratingList, splitRate):
    shuffleLabelList = []
    shuffleRatingList = []
    dataNum = len(idList)
    labelDict = {label[0]:label for label in labelList}
    ratingDict = {rating[0]:rating for rating in ratingList}
    indexList = list(range(len(idList)))
    random.shuffle(indexList)
    for index in indexList:
        shuffleLabelList.append(labelDict[idList[index]])
        shuffleRatingList.append(ratingDict[idList[index]])
    trainLabelList = shuffleLabelList[:int(dataNum*splitRate)]
    testLabelList = shuffleLabelList[int(dataNum*splitRate):]
    trainRatingList = shuffleRatingList[:int(dataNum*splitRate)]
    testRatingList = shuffleRatingList[int(dataNum*splitRate):]

    return trainLabelList, trainRatingList, testLabelList, testRatingList



def splitFeatureLabelData(ratingList, splitRate):
    featList = []
    labelList = []
    for rating in ratingList:
        uid = rating[0]
        ratingData = [r for r in rating[1:]]
        ratingNum = len(ratingData)
        random.shuffle(ratingData)
        featList.append([uid] + ratingData[:int(ratingNum*splitRate)])
        labelList.append([uid] + ratingData[int(ratingNum*splitRate):])
    return featList, labelList


### dump data ###
def dumpRatingFile(filename, ratingList):
    ratingPairList = []
    for rating in ratingList:
        uid = rating[0]
        for mid, r, t in rating[1:]:
            ratingPairList.append([uid, mid, r])
    with open(filename, 'w') as f:
        for pair in ratingPairList:
            uid, mid, r = pair
            print(str(uid)+'\t'+str(mid)+'\t'+str(r), file=f)


def dumpDemoFile(filename, userList):
    with open(filename, 'w') as f:
        for i in range(len(userList)):
            for j in range(len(userList[i])):
                if j < len(userList[i])-1:
                    print(userList[i][j], end='\t', file=f)
                else:
                    print(userList[i][j], file=f)




def make_data():
    ### make MovieLens-1M data ###
    ratingDataSetName = 'ml-1m'
    ratingSplitRate = 0.8
    userSplitRate = 0.7

    trainFeatRatingFile = DATA_PATH_PREFIX + '/' +  ratingDataSetName + '/' + 'train1.rating.feat'
    trainLabelRatingFile = DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'train1.rating.label'
    trainUserFile = DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'train1.user.tab.label'
    testFeatRatingFile = DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'test1.rating.feat'
    testLabelRatingFile = DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'test1.rating.label'
    testUserFile = DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'test1.user.tab.label'

    trainRatingList = []
    testRatingList = []
    userTrainRatingList = []
    userTestRatingList = []
    userTrainDemoList = []
    userTestDemoList = []
    genderList = ['M', 'F']
    ageRangeList = [1, 18, 25, 35, 45, 50, 56]
    occRangeList = list(range(21))

    ## load data ##
    print("load data ...")
    st = time.time()
    ratingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'ratings.dat'
    userFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'users.dat'
    ratingList = loadRatingFile(ratingFileName)
    userList = loadUserFile(userFileName)
    et = time.time()
    print("cost time:", et - st)

    ## process data ##
    print("process data ...")
    st = time.time()
    print("Debug: userList:", userList[0], len(userList[0]), len(userList))
    # extend gender column
    userList = extendColumn(userList, 1, genderList)
    print("Debug: userList:", userList[0], len(userList[0]), len(userList))
    # extend age column
    userList = extendColumn(userList, 2, ageRangeList)
    print("Debug: userList:", userList[0], len(userList[0]), len(userList))
    # extend occupy column
    userList = extendColumn(userList, 3, occRangeList)
    print("Debug: userList:", userList[0], len(userList[0]), len(userList))
    # merge all column
    userList = mergeColumn(userList)
    print("Debug: userList:", userList[0], len(userList[0]), len(userList))
    # collect rating to user
    userRatingList = collectRatingToUser(ratingList)
    et = time.time()
    print("cost time:", et - st)


    ## split data ##
    print("split data ...")
    st = time.time()
    userIdList = [user[0] for user in userList]
#    userDemoList = [user[1:] for user in userList]
    # split data into training, testing set
    trainUserList, trainUserRatingList, testUserList, testUserRatingList = splitTrainTestData(userIdList, userList, userRatingList, userSplitRate)
    print("Debug: trainUserList:", trainUserList[0], len(trainUserList[0]), len(trainUserList))
    print("Debug: testUserList:", testUserList[0], len(testUserList[0]), len(testUserRatingList))
    print("Debug: trainUserRatingList:", trainUserRatingList[0][:10], len(trainUserRatingList[0]), len(trainUserRatingList))
    print("Debug: testUserRatingList:", testUserRatingList[0][:10], len(testUserRatingList[0]), len(testUserRatingList))
    # split rating into feature and label
    trainFeatRatingList, trainLabelRatingList = splitFeatureLabelData(trainUserRatingList, ratingSplitRate)
    testFeatRatingList, testLabelRatingList = splitFeatureLabelData(testUserRatingList, ratingSplitRate)
    print("Debug: trainFeatRatingList:", trainFeatRatingList[0][:10], len(trainFeatRatingList[0]), len(trainFeatRatingList))
    print("Debug: trainLabelRatingList:", trainLabelRatingList[0][:10], len(trainLabelRatingList[0]), len(trainLabelRatingList))
    print("Debug: testFeatRatingList:", testFeatRatingList[0][:10], len(testFeatRatingList[0]), len(testFeatRatingList))
    print("Debug: testLabelRatingList:", testLabelRatingList[0][:10], len(testLabelRatingList[0]), len(testLabelRatingList))
    et = time.time()
    print("cost time:", et - st)

    ## dump data ##
    st = time.time()
    dumpRatingFile(trainFeatRatingFile, trainFeatRatingList)
    dumpRatingFile(trainLabelRatingFile, trainLabelRatingList)
    dumpRatingFile(testFeatRatingFile, testFeatRatingList)
    dumpRatingFile(testLabelRatingFile, testLabelRatingList)
    dumpDemoFile(trainUserFile, trainUserList)
    dumpDemoFile(testUserFile, testUserList)
    et = time.time()
    print("cost time:", et  - st)


def main():
    make_data()



if __name__ == '__main__':
    main()
