import sys
import time
import json

import dataUtils


DATA_PATH_PREFIX = '../data/'
GEN_DATA_PATH_PREFIX = '../gendata/'



def make_ml_cv_files():
    # make MovieLens-100 data files #
    ratingDataSetName = 'ml-100k'
    ratingFileName = 'u.data'
    crossValNum = 5
    labelNum = 3

    trainRatingCVList = [[] for c in range(crossValNum)]
    testRatingCVList = [[] for c in range(crossValNum)]
    trainLabelCVList = [[] for c in range(crossValNum)]
    testLabelCVList = [[] for c in range(crossValNum)]
    userTrRatingCVList = [[] for c in range(crossValNum)]
    userTeRatingCVList = [[] for c in range(crossValNum)]
    userTrDemoCVList = [[] for c in range(crossValNum)]
    userTeDemoCVList = [[] for c in range(crossValNum)]
    trainAllRatingList = []
    testAllRatingList = []
    userTrAllRatingList = []
    userTeAllRatingList = []
    userTrAllDemoList = []
    userTeAllDemoList = []
    userList = []
    userDemoList = []
    ageRangeList = [(0, 17), (18, 35), (36, 65), (66, 100)]
    genderDict = {'M':0, 'F':1}
    occupyDict = {}
    
    print("load data ...")
    st = time.time()
    ## load data
    # load cross validation data
    for c in range(crossValNum):
        loadTrRatingFileName = DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u'+str(c+1)+'.base'
        loadTeRatingFileName = DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u'+str(c+1)+'.test'
        trainRatingCVList[c] = dataUtils.loadRatingList(loadTrRatingFileName)
        testRatingCVList[c] = dataUtils.loadRatingList(loadTeRatingFileName)
    # load user demographic information
    loadUserDemoFileName = DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u.user'
    userDemoList = dataUtils.loadDemoLabelList(loadUserDemoFileName)
    # load occupation attribute
    occupyFileName = DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u.occupation'
    occupyDict, _ = dataUtils.loadDemoCateDict(occupyFileName)
    et = time.time()
    print("cost time:", et - st)

    ## process data
    print("process data ...")
    st = time.time()
    userList = [user[0] for user in userDemoList]
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    # make user demographic labels
    # change age to label value
    dataUtils.rangeValToLabel(userDemoList, ageRangeList, 1)
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    # change gender to label value
    dataUtils.mapColCateToLabel(userDemoList, genderDict, 2)
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    # change occupation to label value
    dataUtils.mapColCateToLabel(userDemoList, occupyDict, 3)
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    # extend to one-hot encoding
    dataUtils.extendColLabelList(userDemoList, len(ageRangeList), 1)
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    dataUtils.extendColLabelList(userDemoList, len(genderDict), 2)
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    dataUtils.extendColLabelList(userDemoList, len(occupyDict), 3)
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    # merge demographic labels
    dataUtils.mergeLabelList(userDemoList, 3)
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    dataUtils.deleteLabelList(userDemoList, 1)      # delete age column
    dataUtils.deleteLabelList(userDemoList, 1)      # delete gender column
    dataUtils.deleteLabelList(userDemoList, 1)      # delete occupation column
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    ## make user cross validation ratings
    for c in range(crossValNum):
        userTrRatingCVList[c] = dataUtils.mergeRatingById(trainRatingCVList[c])
        print("Debug: userTrRatingCVList[c][0]:", c, userTrRatingCVList[c][0])
        userTeRatingCVList[c] = dataUtils.mergeRatingById(testRatingCVList[c])
        print("Debug: userTeRatingCVList[c][0]:", c, userTeRatingCVList[c][0])
        userTrDemoCVList[c] = dataUtils.selectIdByRating(userTrRatingCVList[c], userDemoList)
        print("Debug: userTrDemoCVList[c][0]:", c, userTrDemoCVList[c][0])
        userTeDemoCVList[c] = dataUtils.selectIdByRating(userTeRatingCVList[c], userDemoList)
        print("Debug: userTeDemoCVList[c][0]:", c, userTeDemoCVList[c][0])
        trainAllRatingList += trainRatingCVList[c]
        testAllRatingList += testRatingCVList[c]
        userTrAllRatingList += userTrRatingCVList[c]
        userTeAllRatingList += userTeRatingCVList[c]
        userTrAllDemoList += userTrDemoCVList[c]
        userTeAllDemoList += userTeDemoCVList[c]
    et = time.time()
    print("cost time:", et - st)

    # dump data
    print("dump data ...")
    st = time.time()
    for c in range(crossValNum):
        dumpTrRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u'+str(c+1)+'.rating.train'
        dumpTeRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u'+str(c+1)+'.rating.test'
        dumpTrUserRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u'+str(c+1)+'.rating.user.train'
        dumpTeUserRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u'+str(c+1)+'.rating.user.test'
        dumpTrTabRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u'+str(c+1)+'.rating.user.tab.train'
        dumpTeTabRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u'+str(c+1)+'.rating.user.tab.test'
        dumpTrUserLabelFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u'+str(c+1)+'.label.train'
        dumpTeUserLabelFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u'+str(c+1)+'.label.test'
        dumpTrTabLabelFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u'+str(c+1)+'.label.tab.train'
        dumpTeTabLabelFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u'+str(c+1)+'.label.tab.test'

        dataUtils.dumpRatingList(dumpTrRatingFileName, trainRatingCVList[c], 'user-item')
        dataUtils.dumpRatingList(dumpTeRatingFileName, testRatingCVList[c], 'user-item')
        dataUtils.dumpRatingList(dumpTrUserRatingFileName, userTrRatingCVList[c], 'ml-user-wise')
        dataUtils.dumpRatingList(dumpTeUserRatingFileName, userTeRatingCVList[c], 'ml-user-wise')
        dataUtils.dumpRatingList(dumpTrTabRatingFileName, userTrRatingCVList[c], 'user-wise')
        dataUtils.dumpRatingList(dumpTeTabRatingFileName, userTeRatingCVList[c], 'user-wise')
        dataUtils.dumpLabelList(dumpTrUserLabelFileName, userTrDemoCVList[c], 'ml-user-wise')
        dataUtils.dumpLabelList(dumpTeUserLabelFileName, userTeDemoCVList[c], 'ml-user-wise')
        dataUtils.dumpLabelList(dumpTrTabLabelFileName, userTrDemoCVList[c], 'user-wise')
        dataUtils.dumpLabelList(dumpTeTabLabelFileName, userTeDemoCVList[c], 'user-wise')
    dumpTrRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u_all.rating.train'
    dumpTeRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u_all.rating.test'
    dumpTrUserRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u_all.rating.user.train'
    dumpTeUserRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u_all.rating.user.test'
    dumpTrTabRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u_all.rating.user.tab.train'
    dumpTeTabRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u_all.rating.user.tab.test'
    dumpTrUserLabelFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u_all.label.train'
    dumpTeUserLabelFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u_all.label.test'
    dumpTrTabLabelFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u_all.label.tab.train'
    dumpTeTabLabelFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u_all.label.tab.test'
    dataUtils.dumpRatingList(dumpTrRatingFileName, trainAllRatingList, 'user-item')
    dataUtils.dumpRatingList(dumpTeRatingFileName, testAllRatingList, 'user-item')
    dataUtils.dumpRatingList(dumpTrUserRatingFileName, userTrAllRatingList, 'ml-user-wise')
    dataUtils.dumpRatingList(dumpTeUserRatingFileName, userTeAllRatingList, 'ml-user-wise')
    dataUtils.dumpRatingList(dumpTrTabRatingFileName, userTrAllRatingList, 'user-wise')
    dataUtils.dumpRatingList(dumpTeTabRatingFileName, userTeAllRatingList, 'user-wise')
    dataUtils.dumpLabelList(dumpTrUserLabelFileName, userTrAllDemoList, 'ml-user-wise')
    dataUtils.dumpLabelList(dumpTeUserLabelFileName, userTeAllDemoList, 'ml-user-wise')
    dataUtils.dumpLabelList(dumpTrTabLabelFileName, userTrAllDemoList, 'user-wise')
    dataUtils.dumpLabelList(dumpTeTabLabelFileName, userTeAllDemoList, 'user-wise')
    et = time.time()
    print("cost time:", et - st)




def make_ml100k_train_test_data():
    # MovieLens-100 data files #
    ratingDataSetName = 'ml-100k'
    ratingSplitRate = 0.8
    userSplitRate = 0.7

    trainRatingList = []
    testRatingList = []
    userTrainRatingList = []
    userTestRatingList = []
    userTrainDemoList = []
    userTestDemoList = []
    ageRangeList = [(0, 17), (18, 35), (36, 65), (66, 100)]
    genderDict = {'M':0, 'F':1}
    occupyDict = {}

    ## load data
    print("load data ...")
    st = time.time()
    ratingFileName = DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u.data'
    userDemoFileName = DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u.user'
    occupyFileName = DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'u.occupation'
    ratingList = dataUtils.loadRatingList(ratingFileName)
    userDemoList = dataUtils.loadDemoLabelList(userDemoFileName)
    occupyDict, _ = dataUtils.loadDemoCateDict(occupyFileName)
    et = time.time()
    print("cost time:", et - st)

    ## process data
    print("process data ...")
    st = time.time()
    # make user demographic labels 
    userList = [user[0] for user in userDemoList]
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    # change age to label value
    dataUtils.rangeValToLabel(userDemoList, ageRangeList, 1)
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    # change gender to label value
    dataUtils.mapColCateToLabel(userDemoList, genderDict, 2)
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    # change occupation to label value
    dataUtils.mapColCateToLabel(userDemoList,occupyDict, 3)
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))    
    # extend to one-hot encoding
    dataUtils.extendColLabelList(userDemoList, len(ageRangeList), 1)
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    dataUtils.extendColLabelList(userDemoList, len(genderDict), 2)
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    dataUtils.extendColLabelList(userDemoList, len(occupyDict), 3)
    # merge demographic labels
    dataUtils.mergeLabelList(userDemoList, 3)
    dataUtils.deleteLabelList(userDemoList, 1)          # delete age column
    dataUtils.deleteLabelList(userDemoList, 1)          # delete gender column
    dataUtils.deleteLabelList(userDemoList, 1)          # delete occupation column
    print("Debug: userDemoList[0]:", userDemoList[0], len(userDemoList))
    # split data to feature label rating
    featRatingList, labelRatingList = dataUtils.splitPairRating(ratingList, ratingSplitRate, 'userTimeSequence')
    # merge data to users
    userFeatRatingList = dataUtils.mergeRatingById(featRatingList)
    userLabelRatingList = dataUtils.mergeRatingById(labelRatingList)
    print("Debug: userFeatRatingList[0]:", userFeatRatingList[0][0], len(userFeatRatingList))
    # split data to train test users
    userTrFeatRatingList, userTeFeatRatingList = dataUtils.splitRowRating(userFeatRatingList, userSplitRate, 'random')
    print("Debug: userTrFeatRatingList[0]:", userTrFeatRatingList[0], len(userTrFeatRatingList))
    # select user
    userTrLabelRatingList = dataUtils.selectIdByRating(userTrFeatRatingList, userLabelRatingList)
    userTeLabelRatingList = dataUtils.selectIdByRating(userTeFeatRatingList, userLabelRatingList)
    print("Debug: userTrLabelRatingList[0]:", userTrLabelRatingList[0], len(userTrLabelRatingList))
    userTrDemoList = dataUtils.selectIdByRating(userTrFeatRatingList, userDemoList)
    userTeDemoList = dataUtils.selectIdByRating(userTeFeatRatingList, userDemoList)
    # make pair list
    trFeatRatingList = dataUtils.divRatingFromId(userTrFeatRatingList)
    teFeatRatingList = dataUtils.divRatingFromId(userTeFeatRatingList)
    trLabelRatingList = dataUtils.divRatingFromId(userTrLabelRatingList)
    teLabelRatingList = dataUtils.divRatingFromId(userTeLabelRatingList)
    et = time.time()
    print("cost time:", et - st)

    # dump data
    print("dump data ...")
    st = time.time()
    dumpTrFeatRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'train1.rating.feat'
    dumpTrLabelRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'train1.rating.label'
    dumpTeFeatRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'test1.rating.feat'
    dumpTeLabelRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'test1.rating.label'

    dumpTrUserFeatRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'train1.user.rating.feat'
    dumpTrUserLabelRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'train1.user.rating.label'
    dumpTeUserFeatRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'test1.user.rating.feat'
    dumpTeUserLabelRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'test1.user.rating.label'

    dumpTrTabFeatRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'train1.user.tab.rating.feat'
    dumpTrTabLabelRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'train1.user.tab.rating.label'
    dumpTeTabFeatRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'test1.user.tab.rating.feat'
    dumpTeTabLabelRatingFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'test1.user.tab.rating.label'

    dumpTrUserDemoFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'train1.user.label'
    dumpTeUserDemoFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'test1.user.label'

    dumpTrTabDemoFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'train1.user.tab.label'
    dumpTeTabDemoFileName = GEN_DATA_PATH_PREFIX + '/' + ratingDataSetName + '/' + 'test1.user.tab.label'

    dataUtils.dumpRatingList(dumpTrFeatRatingFileName, trFeatRatingList, 'user-item')
    dataUtils.dumpRatingList(dumpTrLabelRatingFileName, trLabelRatingList, 'user-item')
    dataUtils.dumpRatingList(dumpTeFeatRatingFileName, teFeatRatingList, 'user-item')
    dataUtils.dumpRatingList(dumpTeLabelRatingFileName, teLabelRatingList, 'user-item')
    
    dataUtils.dumpRatingList(dumpTrUserFeatRatingFileName, userTrFeatRatingList, 'ml-user-wise')
    dataUtils.dumpRatingList(dumpTrUserLabelRatingFileName, userTrLabelRatingList, 'ml-user-wise')
    dataUtils.dumpRatingList(dumpTeUserFeatRatingFileName, userTeFeatRatingList, 'ml-user-wise')
    dataUtils.dumpRatingList(dumpTeUserLabelRatingFileName, userTeLabelRatingList, 'ml-user-wise')

    dataUtils.dumpRatingList(dumpTrTabFeatRatingFileName, userTrFeatRatingList, 'user-wise')
    dataUtils.dumpRatingList(dumpTrTabLabelRatingFileName, userTrLabelRatingList, 'user-wise')
    dataUtils.dumpRatingList(dumpTeTabFeatRatingFileName, userTeFeatRatingList, 'user-wise')
    dataUtils.dumpRatingList(dumpTeTabLabelRatingFileName, userTeLabelRatingList, 'user-wise')

    dataUtils.dumpLabelList(dumpTrUserDemoFileName, userTrDemoList, 'ml-user-wise')
    dataUtils.dumpLabelList(dumpTeUserDemoFileName, userTeDemoList, 'ml-user-wise')

    dataUtils.dumpLabelList(dumpTrTabDemoFileName, userTrDemoList, 'user-wise')
    dataUtils.dumpLabelList(dumpTeTabDemoFileName, userTeDemoList, 'user-wise')
    et = time.time()
    print("cost time:", et - st)


def make_yelp_train_test_data():
    # yelp academic dataset files #
    datasetName = 'yelp_dataset'
    revFileName = 'yelp_academic_dataset_review.json'
    busFileName = 'yelp_academic_dataset_business.json'
    usrFileName = 'yelp_academic_dataset_user.json'
    ratingSplitRate = 0.8
    busSplitRate = 0.7
    selBusNum = 30000
    selUsrNum = 100000
    selCateNum = 10

    trainRatingList = []
    testRatingList = []
    busTrainRatingList = []
    busTestRatingList = []
    busTrainDemoList = []
    busTestDemoList = []

    ## load data ##
    print("load data ...")
    st = time.time()
    revFileName = DATA_PATH_PREFIX + datasetName + '/' + revFileName
    busFileName = DATA_PATH_PREFIX + datasetName + '/' + busFileName
    usrFileName = DATA_PATH_PREFIX + datasetName + '/' + usrFileName
    revJsonList = dataUtils.loadJSONList(revFileName)
    busJsonList = dataUtils.loadJSONList(busFileName)
    usrJsonList = dataUtils.loadJSONList(usrFileName)
    et = time.time()
    print("cost time:", et - st)
    ## process data ##
    print("process data ...")
    st = time.time()
    # get info from json
    ratingList = dataUtils.getJSONKeyValues(revJsonList, ['business_id', 'user_id', 'stars', 'date'])
    busIdList = dataUtils.getJSONKeyValues(busJsonList, ['business_id'])
    usrIdList = dataUtils.getJSONKeyValues(usrJsonList, ['user_id'])
    busCateList = dataUtils.getJSONKeyValues(busJsonList, ['business_id', 'categories'])
    del revJsonList
    del busJsonList
    del usrJsonList
    ratingList = [[ratingList[i][j] if j != 3 else int(ratingList[i][j].replace('-', '')) for j in range(4)] for i in range(len(ratingList))] # convert date to a comparable number
    busIdList = [busIdList[i][0] for i in range(len(busIdList))]
    usrIdList = [usrIdList[i][0] for i in range(len(usrIdList))]
    busCateList = [[busCateList[i][0]] + busCateList[i][1] if busCateList[i][1] != None else [busCateList[i][0]] for i in range(len(busCateList))]
    print("Debug: ratingList[0]:", ratingList[0])
    print("Debug: busCateList[0]:", busCateList[0])
    cateList = [busCateList[i][j] for i in range(len(busCateList)) for j in range(1, len(busCateList[i]))]
    cateIdList = list(set(cateList))
    # select the categories with highest counts
    cateCountList = dataUtils.countEleById(cateList, cateIdList)
    cateCountItems = [(cateIdList[i], cateCountList[i]) for i in range(len(cateIdList))]
    cateCountItems = sorted(cateCountItems, key=lambda x:x[1], reverse=True)
    selCateIdList = [cateCountItems[i][0] for i in range(selCateNum)]
    del cateIdList
    print("Debug: selected categories:", selCateIdList, "id num:", len(selCateIdList))
    # select the businesses with at least 1 of all selected categories
    selBusCateList = dataUtils.selectElementByAllCate(busCateList, selCateIdList, 'atLeastOne')
    selBusIdList = [selBusCateList[i][0] for i in range(len(selBusCateList))]
    del busIdList
    print("Debug: selected business ids num:", len(selBusIdList))
    # select the ratings with selected bussiness ids
    selRatingList = dataUtils.selectRatingByColId(ratingList, selBusIdList, 0)
    del ratingList
    # select the businesses with highest counts in reviews (in selRatingList)
    busIdList = selBusIdList
    busCountList = dataUtils.countRatingByColId(selRatingList, busIdList, 0)
    busCountItems = [(busIdList[i], busCountList[i]) for i in range(len(busIdList))]
    busCountItems = sorted(busCountItems, key=lambda x:x[1], reverse=True)
    selBusIdList = [busCountItems[i][0] for i in range(selBusNum)]
    del busIdList
    print("Debug: selected business ids num:", len(selBusIdList))
    # select the ratings with selected business ids
    ratingList = selRatingList
    selRatingList = dataUtils.selectRatingByColId(ratingList, selBusIdList, 0)
    del ratingList
    # select the users with highest counts of reviews (in selRatingList)
    usrCountList = dataUtils.countRatingByColId(selRatingList, usrIdList, 1)
    usrCountItems = [(usrIdList[i], usrCountList[i]) for i in range(len(usrIdList))]
    usrCountItems = sorted(usrCountItems, key=lambda x:x[1], reverse=True)
    selUsrIdList = [usrCountItems[i][0] for i in range(selUsrNum)]
    del usrIdList
    print("Debug: selected user ids num:", len(selUsrIdList))
    # select the ratings with selected user ids
    ratingList = selRatingList
    selRatingList = dataUtils.selectRatingByColId(ratingList, selUsrIdList, 1)
    del ratingList
    print("Debug: selRatingList[0]:", selRatingList[0], len(selRatingList))
    # make id map
    busIdDict, revBusIdDict = dataUtils.makeIdDict(selBusIdList)
    usrIdDict, revUsrIdDict = dataUtils.makeIdDict(selUsrIdList)
    cateIdDict, revCateIdDict = dataUtils.makeIdDict(selCateIdList)
    print("Debug: cateIdDict:", cateIdDict)
    # select the categories with business ids
    selBusCateList = dataUtils.selectLabelByColId(busCateList, selBusIdList, 0)
    del busCateList
    print("Debug: selBusCateList[0]:", selBusCateList[0], len(selBusCateList))
    # change business, users id to id numbers
    dataUtils.mapColCateToLabel(selBusCateList, busIdDict, 0)
    dataUtils.mapColCateToLabel(selRatingList, busIdDict, 0)
    dataUtils.mapColCateToLabel(selRatingList, usrIdDict, 1)
    print("Debug: selRatingList[0]:", selRatingList[0])
    print("Debug: selBusCateList[0]:", selBusCateList[0])
    # change categories to category ids
    dataUtils.mapAllCateToLabel(selBusCateList, cateIdDict)
    print("Debug: selBusCateList[0]:", selBusCateList[0])
    # extend to binary one-hot vectors and concatenate them
    dataUtils.extendAllColBinLabelList(selBusCateList, selCateNum)
    print("Debug: selBusCateList[0]:", selBusCateList[0])
    # check business categories list
    if not dataUtils.checkOneHotList(selBusCateList):
        print("Error: Business category list is not a one-hot list")
        assert False
    # merge rating to business
    busRatingList = dataUtils.mergeRatingById(selRatingList)
    print("Debug: busRatingList[0]:", busRatingList[0], len(busRatingList))
    # split data to train test users
    busTrRatingList, busTeRatingList = dataUtils.splitRowRating(busRatingList, busSplitRate, 'random')
    print("Debug: busTrRatingList[0]:", busTrRatingList[0], len(busTrRatingList))
    print("Debug: busTeRatingList[0]:", busTeRatingList[0], len(busTeRatingList))
    # split data to feature label rating
    busTrFeatRatingList, busTrLabelRatingList = dataUtils.splitColRating(busTrRatingList, ratingSplitRate, 'userTimeSequence')
    busTeFeatRatingList, busTeLabelRatingList = dataUtils.splitColRating(busTeRatingList, ratingSplitRate, 'userTimeSequence')
    print("Debug: busTrFeatRatingList[0]:", busTrFeatRatingList[0], len(busTrFeatRatingList))
    print("Debug: busTrLabelRatingList[0]:", busTrLabelRatingList[0], len(busTrLabelRatingList))
    print("Debug: busTeFeatRatingList[0]:", busTeFeatRatingList[0], len(busTeFeatRatingList))
    print("Debug: busTeLabelRatingList[0]:", busTeLabelRatingList[0], len(busTeLabelRatingList))
    busTrCateList = dataUtils.selectIdByRating(busTrFeatRatingList, selBusCateList)
    busTeCateList = dataUtils.selectIdByRating(busTeFeatRatingList, selBusCateList)
    print("Debug: busTrCateList[0]:", busTrCateList[0], len(busTrCateList))
    print("Debug: busTeCateList[0]:", busTeCateList[0], len(busTeCateList))
    # make pair list
    trFeatRatingList = dataUtils.divRatingFromId(busTrFeatRatingList)
    teFeatRatingList = dataUtils.divRatingFromId(busTeFeatRatingList)
    trLabelRatingList = dataUtils.divRatingFromId(busTrLabelRatingList)
    teLabelRatingList =  dataUtils.divRatingFromId(busTeLabelRatingList)
    et = time.time()
    print("cost time:", et - st)

    # dumpy data
    print("dump data ...")
    st = time.time()
    dumpTrFeatRatingFileName = GEN_DATA_PATH_PREFIX + datasetName + '/' + 'train1.rating.feat'
    dumpTrLabelRatingFileName = GEN_DATA_PATH_PREFIX + datasetName + '/' + 'train1.rating.label'
    dumpTeFeatRatingFileName = GEN_DATA_PATH_PREFIX  + datasetName + '/' + 'test1.rating.feat'
    dumpTeLabelRatingFileName = GEN_DATA_PATH_PREFIX + datasetName + '/' + 'test1.rating.label'

    dumpTrTabBusFeatRatingFileName = GEN_DATA_PATH_PREFIX + datasetName + '/' + 'train1.bus.tab.rating.feat'
    dumpTrTabBusLabelRatingFileName = GEN_DATA_PATH_PREFIX + datasetName + '/' + 'train1.bus.tab.rating.label'
    dumpTeTabBusFeatRatingFileName = GEN_DATA_PATH_PREFIX + datasetName + '/' + 'test1.bus.tab.rating.feat'
    dumpTeTabBusLabelRatingFileName = GEN_DATA_PATH_PREFIX + datasetName + '/' + 'test1.bus.tab.rating.label'

    dumpTrTabBusCateFileName = GEN_DATA_PATH_PREFIX + datasetName + '/' + 'train1.bus.tab.label'
    dumpTeTabBusCateFileName = GEN_DATA_PATH_PREFIX + datasetName + '/' + 'test1.bus.tab.label'

    dataUtils.dumpRatingList(dumpTrFeatRatingFileName, trFeatRatingList, 'user-item')
    dataUtils.dumpRatingList(dumpTrLabelRatingFileName, trLabelRatingList, 'user-item')
    dataUtils.dumpRatingList(dumpTeFeatRatingFileName, teFeatRatingList, 'user-item')
    dataUtils.dumpRatingList(dumpTeLabelRatingFileName, teLabelRatingList, 'user-item')

    dataUtils.dumpRatingList(dumpTrTabBusFeatRatingFileName, busTrFeatRatingList, 'user-wise')
    dataUtils.dumpRatingList(dumpTrTabBusLabelRatingFileName, busTrLabelRatingList, 'user-wise')
    dataUtils.dumpRatingList(dumpTeTabBusFeatRatingFileName, busTeFeatRatingList, 'user-wise')
    dataUtils.dumpRatingList(dumpTeTabBusLabelRatingFileName, busTeLabelRatingList, 'user-wise')

    dataUtils.dumpLabelList(dumpTrTabBusCateFileName, busTrCateList, 'user-wise')
    dataUtils.dumpLabelList(dumpTeTabBusCateFileName, busTeCateList, 'user-wise')
    et = time.time()
    print("cost time:", et - st)
    





def main():
    # make MovieLens 100K train test data
    #make_ml100k_train_test_data()

    # make yelp academic dataset train test data
    make_yelp_train_test_data()







if __name__ == '__main__':
    main()
