# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 08:33:45 2023

@author: yjy
"""

import random
import pandas as pd
import numpy as np

#calculate Mahalanobis distance
def calcMaDis(dataSet, centroids, k, inv_cov):
    claMaList = []
    for data in dataSet:
        d1=[]
        for i in range(k):
            delta = np.array(data) - np.array(centroids[i])
            d=np.sqrt(np.dot(np.dot(delta,inv_cov[i]),delta.T))
            d1.append(d)
        claMaList.append(d1)
    claMaList = np.array(claMaList) 
    return claMaList

#calculate centroids
def classify(dataSet, centroids, k , inv_cov):
    # calculate Mahalanobis distance between sample and centroids 
    clalist = calcMaDis(dataSet, centroids, k, inv_cov)
    # group and calculate the new centroids 
    minDistIndices = np.argmin(clalist, axis=1)    #axis=1 
    newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean() 
    newCentroids = newCentroids.values
    
    # calculate the change 
    changed = newCentroids - centroids
    
    return changed, newCentroids

#K-means
def kmeans(dataSet, k, inv_cov):
    # 随机取质心
    centroids = random.sample(dataSet, k)
    
    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k, inv_cov)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k, inv_cov)
    
    centroids = sorted(newCentroids.tolist())   #tolist()将矩阵转换成列表 sorted()排序
    
    # 根据质心计算每个集群
    
    clalist = calcMaDis(dataSet, centroids, k, inv_cov) #调用马氏距离
    minDistIndices = np.argmin(clalist, axis=1)  
    type = minDistIndices +1 
    
    return type





###test###3


# =============================================================================
# def createDataSet():
#     return [[1, 1, 2], [6, 4, 5], [1, 2, 1], [2, 1 ,1], [6, 3, 5], [5, 4 ,6]]
# 
# dataset = createDataSet()
# 
# centroids = random.sample(dataset, 2)
# 
# temp = np.array([[1,0,0],[0,1,0],[0,0,1]])
# temp2 = temp
# inv_cov = [temp,temp]
# 
# type  = kmeans(dataset, 2, inv_cov)
# 
# print('集群为：%s' % type)
# =============================================================================

