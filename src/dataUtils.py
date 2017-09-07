import random
import json



# load rating from file with movieLens format
def loadRatingList(filename):
    ratingList = []
    with open(filename, 'r') as f:
        for line in f:
            uid, iid, r, t = line.strip().split('\t')
            uid, iid, r, t = int(uid), int(iid), float(r), int(t)
            ratingList.append([uid, iid, r, t])
    return ratingList



# load user demographic information from file with movieLens format
def loadDemoLabelList(filename):
    labelList = []
    with open(filename, 'r') as f:
        for line in f:
            labels = line.strip().split('|')
            uid, age, gender, occ, _ = labels
            uid, age, gender, occ = int(uid), int(age), gender, occ
            labelList.append([uid, age, gender, occ])
    return labelList


# load demographic values from movieLens files
def loadDemoCateDict(filename):
    cateDict = {}
    revDict = {}
    index = 0
    with open(filename, 'r') as f:
        for line in f:
            cate = line.strip()
            cateDict[cate] = index
            revDict[index] = cate
            index += 1
    return cateDict, revDict


# load files with json-formats lines
def loadJSONList(filename):
    jsonList = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            jsonList.append(json.loads(line))
    return jsonList


# dump demographic to file with tab-separation
def dumpLabelList(filename, labelList, fileFormat):
    if fileFormat == 'user-wise':
        with open(filename, 'w') as f:
            for labels in labelList:
                user = labels[0]
                print(user, end='\t', file=f)
                for l in range(len(labels[1])):
                    if l < len(labels[1])-1:
                        print(labels[1][l], end='\t', file=f)
                    else:
                        print(labels[1][l], end='', file=f)
                print('', file=f)
    elif fileFormat == 'ml-user-wise':
        with open(filename, 'w') as f:
            for labels in labelList:
                user = labels[0]
                print(str(user)+'|', end='', file=f)
                for l in range(len(labels[1])):
                    if l < len(labels[1])-1:
                        print(str(labels[1][l])+'|', end='', file=f)
                    else:
                        print(labels[1][l], end='', file=f)
                print('', file=f)
    else:
        print("Error: unknown dumping file format")


# dump rating to file
def dumpRatingList(filename, ratingList, fileFormat):
    if fileFormat == 'user-item':
        with open(filename, 'w') as f:
            for rating in ratingList:
                uid, iid, r, t = rating
                print(str(uid)+'\t'+str(iid)+'\t'+str(r), file=f)
    elif fileFormat == 'user-wise':
        with open(filename, 'w') as f:
            for rating in ratingList:
                user = rating[0]
                print(user, end='\t', file=f)
                for i in range(1,len(rating)):
                    iid, r, t = rating[i]
                    if i < len(rating)-1:
                        print(str(iid)+'\t'+str(r), end='', file=f)
                    else:
                        print(str(iid)+'\t'+str(r)+'\t', end='', file=f)
                print('', file=f)
    elif fileFormat == 'ml-user-wise':
        with open(filename, 'w') as f:
            for rating in ratingList:
                user = rating[0]
                print(str(user)+'|', end='', file=f)
                for i in range(1, len(rating)):
                    iid, r, t = rating[i]
                    if i < len(rating)-1:
                        print(str(iid)+'|'+str(r)+'|', end='', file=f)
                    else:
                        print(str(iid)+'|'+str(r), end='', file=f)
                print('', file=f)
    else:
        print("Error: unknown dumping file format")



# get the values in a json list of keys of a key list 
def getJSONKeyValues(jsonList, keysList):
    jsonValuesList = []
    for i in range(len(jsonList)):
        valuesList = []
        for key in keysList:
            valuesList.append(jsonList[i][key])
        jsonValuesList.append(valuesList)
    return jsonValuesList



# map elements of a list to a dictionary and map back reversely
def makeIdDict(elementList):
    idDict = {}
    revDict = {}
    index = 0
    for i in range(len(elementList)):
        if elementList[i] not in idDict:
            idDict[elementList[i]] = index
            revDict[index] = elementList[i]
            index += 1
    return idDict, revDict



# count elements of elementList by id in idList
def countEleById(elementList, idList):
    countList = []
    countDict = {idList[i]:0 for i in range(len(idList))}
    for i in range(len(elementList)):
        if elementList[i] in countDict:
            countDict[elementList[i]] += 1
    for i in range(len(idList)):
        countList.append(countDict[idList[i]])
    del countDict
    return countList



# count the number of rating in ratingList by id in idCol of idList
def countRatingByColId(ratingList, idList, idCol):
    countList = []
    countDict = {idList[i]:0 for i in range(len(idList))}
    for i in range(len(ratingList)):
        if ratingList[i][idCol] in countDict:
            countDict[ratingList[i][idCol]] += 1
    for i in range(len(idList)):
        countList.append(countDict[idList[i]])
    del countDict
    return countList



def mapAllCateToLabel(labelList, cateDict):
    for i in range(len(labelList)):
        # not change column 0 (id)
        for j in range(1, len(labelList[i])):
            # map if the label in dict
            if labelList[i][j] in cateDict:
                labelList[i][j] = cateDict[labelList[i][j]]
            else:
                labelList[i][j] = None
    return labelList




# map the categority in a specific column to the label number by using mapping dictionary
def mapColCateToLabel(labelList, cateDict, demoCol):
    for i in range(len(labelList)):
        labelList[i][demoCol] = cateDict[labelList[i][demoCol]]
    return labelList



# split numerical value to label number by using spliting range
def rangeValToLabel(labelList, rangeList, demoCol):
    for i in range(len(labelList)):
        for r in range(len(rangeList)):
            if labelList[i][demoCol] >= rangeList[r][0] and labelList[i][demoCol] <= rangeList[r][1]:
                labelList[i][demoCol] = r
                break
    return labelList


# extend all labels to a one-hot vector
def extendAllLabelList(labelList, labelNum):
    for l in range(len(labelList)):
        newExtDemo = [0 for i in range(labelNum)]
        # not change column 0 (id)
        Id = labelList[l][0]
        for j in range(1,len(labelList[l])):
#            print("Debug: labelList", l, j, ":", labelList[l][j])
            if labelList[l][j] != None and labelList[l][j] < labelNum:
                newExtDemo[labelList[l][j]] = 1
        labelList[l] = [Id, newExtDemo]
    return labelList


# extend the label in a specific column from numbers to a one-hot vector
def extendColLabelList(labelList, labelNum, demoCol):
    for l in range(len(labelList)):
        newExtDemo = [0 for i in range(labelNum)]
        newExtDemo[labelList[l][demoCol]] = 1
        labelList[l][demoCol] = newExtDemo
    return labelList


# extend the label in each column to a one-hot binary vector and concatenate them
def extendAllColBinLabelList(labelList, labelNum):
    for l in range(len(labelList)):
        # not change column 0 (id)
        Id = labelList[l][0]
        # [0, 1]: without label
        # [1, 0]: with label
        newExtLabel = [0 if i % 2 == 0 else 1 for i in range(labelNum*2)]
        for j in range(1,len(labelList[l])):
            if labelList[l][j] != None and labelList[l][j] < labelNum:
                newExtLabel[labelList[l][j]*2] = 1
                newExtLabel[labelList[l][j]*2+1] = 0
        labelList[l] = [Id, newExtLabel]
    return labelList



# merge different demographic label vectors to one label vectors
def mergeLabelList(labelList, demoNum):
    for l in range(len(labelList)):
        labelList[l].append([])
        newMrgDemo = []
        for d in range(1, demoNum+1):
            newMrgDemo += labelList[l][d]
        labelList[l][demoNum+1] = newMrgDemo
    return labelList



# merge ratings with the same user id
def mergeRatingById(ratingList):
    mergeRatingList = []
    userDict = {}
    index = 0
    for i in range(len(ratingList)):
        uid, iid, r, t = ratingList[i]
        if uid not in userDict:
            userDict[uid] = index
            mergeRatingList.append([uid])
            index += 1
    for i in range(len(ratingList)):
        uid, iid, r, t = ratingList[i]
        mergeRatingList[userDict[uid]].append([iid, r, t])
    return mergeRatingList



# divide ratings from user ids 
def divRatingFromId(userRatingList):
    divRatingList = []
    for u in range(len(userRatingList)):
#        print("Debug: userRatingList[u]:", u, userRatingList[u])
        uid = userRatingList[u][0]
        for i in range(1, len(userRatingList[u])):
            iid, r, t = userRatingList[u][i]
            divRatingList.append([uid, iid, r, t])
    return divRatingList



# delete the column of demographic label
def deleteLabelList(labelList, demoCol):
    for l in range(len(labelList)):
        del labelList[l][demoCol]



# select the label in labelList that the id in idCol is in idList
def selectLabelByColId(labelList, idList, idCol):
    selectLabelList = []
    idSet = set(idList)
    for i in range(len(labelList)):
        if labelList[i][idCol] in idSet:
            selectLabelList.append(labelList[i])
    return selectLabelList




# select the rating in ratingList that the id in idCol is in idList
def selectRatingByColId(ratingList, idList, idCol):
    selectRatingList = []
    idSet = set(idList)
    for i in range(len(ratingList)):
        if ratingList[i][idCol] in idSet:
            selectRatingList.append(ratingList[i])
    return selectRatingList




# select the users that appear in rating list
def selectIdByRating(userRatingList, userAttrList):
    selectUserAttrList = []
    ratingUserNum = len(userRatingList)
    ratingUserDict = {userRatingList[i][0]:i for i in range(ratingUserNum)}
    selectUserAttrList = [[] for i in range(ratingUserNum)]
    for i in range(len(userAttrList)):
        if userAttrList[i][0] in ratingUserDict:
#            print("Debug: userAttr:", userAttr, "ratingUserDict[0]:", list(ratingUserDict.keys())[0])
            selectUserAttrList[ratingUserDict[userAttrList[i][0]]] += userAttrList[i]
    for i in range(len(selectUserAttrList)):
        if selectUserAttrList[i] == []:
            print("Debug: select user attr empty:", i)
    return selectUserAttrList



def selectElementByAllCate(elementList, cateIdList, selectRule):
    selectElementList = None
    cateIdSet = set(cateIdList)
    if selectRule == 'atLeastOne':
        selectElementList = []
        for element in elementList:
            isSelected = False
            Id, cates = element[0], element[1:]
            for cate in cates:
                if cate in cateIdSet:
                    isSelected = True
                    break
            if isSelected:
                selectElementList.append(element)
    else:
        print("Error: Unknown select rule")
    return selectElementList



def splitPairRating(ratingList, splitRate, splitRule):
    trRatingList = None
    teRatingList = None
    if splitRule == 'timeSequence':
        ratingNum = len(ratingList)
        timeRatingList = ratingList
        timeRatingList = sorted(timeRatingList, key=lambda x: x[3])
        trRatingList = timeRatingList[:int(ratingNum*splitRate)]
        teRatingList = timeRatingList[int(ratingNum*splitRate):ratingNum]
    elif splitRule == 'userTimeSequence':
        trRatingList = []
        teRatingList = []
        userDict = {}
        for rating in ratingList:
            if rating[0] not in userDict:
                userDict[rating[0]] = []
            userDict[rating[0]].append(rating)
        for user in userDict:
            ratingNum = len(userDict[user])
            userDict[user] = sorted(userDict[user], key=lambda x: x[3])
            trRatingList += userDict[user][:int(ratingNum*splitRate)]
            teRatingList += userDict[user][int(ratingNum*splitRate):ratingNum]
    elif splitRule == 'timeRound':
        pass
    else:
        print("Error: unknown split rule")

    return trRatingList, teRatingList



# split data to training and testing data
def splitRowRating(ratingList, splitRate, splitRule):
    trRatingList = None
    teRatingList = None
    if splitRule == 'random':
        trRatingList = []
        teRatingList = []
        userNum = len(ratingList)
        randIndices = list(range(userNum))
        random.shuffle(randIndices)
        for i in range(int(userNum*splitRate)):
            trRatingList.append(ratingList[randIndices[i]])
        for i in range(int(userNum*splitRate), userNum):
            teRatingList.append(ratingList[randIndices[i]])
        
    elif splitRule == 'ratingBalance':
        pass
    else:
        print("Error: unknown split rule")

    return trRatingList, teRatingList



# split data to feature and label (used to test) rating
def splitColRating(ratingList, splitRate, splitRule):
    trRatingList = None
    teRatingList = None
    if splitRule == 'userTimeSequence':
        trRatingList = []
        teRatingList = []
        for rating in ratingList:
            Id, rating = rating[0], rating[1:]
            ratingNum = len(rating)
            rating = sorted(rating, key=lambda x: x[2])  # rating: [[iid, r, t]]
            trRatingList.append([Id] + rating[:int(ratingNum*splitRate)])
            teRatingList.append([Id] + rating[int(ratingNum*splitRate):])
    elif splitRule == 'timeRound':
        pass
    else:
        print("Error: unknown split rule")

    return trRatingList, teRatingList



def checkOneHotList(elementList):
    isOneHot = True
    for i in range(len(elementList)):
        for j in range(1, len(elementList[i][1])):
            if elementList[i][1][j] != 0 and elementList[i][1][j] != 1:
                print("Not one-hot:", i, j, elementList[i][1][j])
                isOneHot = False
    return isOneHot
