import sys
import time
import json

import dataUtils

DATA_PATH_PREFIX = '../data/'
GEN_DATA_PATH_PREFIX =  '../gendata/'
ANALYSIS_PATH_PREFIX = '../analysis/'


def count_user_number():
    datasetName = 'yelp_dataset'
    usrFileName = 'yelp_academic_dataset_user.json'
    alyFileName = 'yelp_user_count.txt'
    usrFilePath = DATA_PATH_PREFIX + datasetName + '/' + usrFileName
    alyFilePath = ANALYSIS_PATH_PREFIX + alyFileName
    print("load data ...")
    st = time.time()
    userList = dataUtils.loadJSONList(usrFilePath)
    et = time.time()
    print("cost time:", et - st)
    print("user number:", len(userList))


def analyze_yelp_business_category():
    datasetName = 'yelp_dataset'
    busFileName = 'yelp_academic_dataset_business.json'
    alyFileName = 'yelp_business_category_count.txt'
    busFilePath = DATA_PATH_PREFIX + datasetName + '/' + busFileName
    alyFilePath = ANALYSIS_PATH_PREFIX + alyFileName
    print("load data ...")
    st = time.time()
    businessList = dataUtils.loadJSONList(busFilePath)
    et = time.time()
    print("cost time:", et - st)
    busCountDict = {}
    cateCountDict = {}
    print("analyze data ...")
    st = time.time()
    for i in range(len(businessList)):
        busId = businessList[i]['business_id']
        busCate = businessList[i]['categories']
        if busCate != None:
            busCountDict[busId] = len(busCate)
            for cate in busCate:
                if cate not in cateCountDict:
                    cateCountDict[cate] = 0
                cateCountDict[cate] += 1
        else:
            busCountDict[busId] = 0

    busCountItems = list(busCountDict.items())
    cateCountItems = list(cateCountDict.items())
    busCountItems = sorted(busCountItems, key=lambda x: x[1], reverse=True)
    cateCountItems = sorted(cateCountItems, key=lambda x: x[1], reverse=True)
    maxBusCount = max(busCountDict.values())
    minBusCount = min(busCountDict.values())
    avgBusCount = sum(busCountDict.values()) / len(busCountDict.keys())
    maxCateCount = max(cateCountDict.values())
    minCateCount = min(cateCountDict.values())
    avgCateCount = sum(cateCountDict.values()) / len(cateCountDict.keys())
    print("business number:", len(busCountDict))
    print("top 10 business with categories count")
    for i in range(10):
        print('id:', busCountItems[i][0], 'count:', busCountItems[i][1])
    print("business count:", "max:", maxBusCount, "min:", minBusCount, "mean:", avgBusCount)
    print("category number:", len(cateCountDict))
    print("top 50 category with count")
    for i in range(50):
        print('cate:', cateCountItems[i][0], 'count:', cateCountItems[i][1])
    print("category count:", "max:", maxCateCount, "min:", minBusCount, "mean:", avgBusCount)
    et = time.time()
    print("cost time:", et - st)
        
        




def main():
    count_user_number()
    #analyze_yelp_business_category()


if __name__ == '__main__':
    main()
