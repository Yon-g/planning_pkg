
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import time
import random

epsilon = 3
min_samples = 2


# Generate data
# map_data_txt = pd.read_csv("/home/yeah/catkin_ws/src/lidar_to_planning/UTM_sub.txt", sep=',', encoding='utf-8')
# UTMmap_arr= map_data_txt.to_numpy()


# input : 2차원 numpy arr (클러스트링 할 점들)
# output : 1차원 numpy arr (클러스터링 된 점들의 중앙값)
def np_dbscan(np_arr):
    c_time = time.time()

    # Compute DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(np_arr)
    labels = db.labels_
    # db의 레이블을 사용하여 bool 배열을 생성, outlier는 False로 변경
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True

    # # 레이블에 있는 클러스터 수, 노이즈가 있을 경우 무시
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise = list(labels).count(-1)
    # print('n_clusters_->', n_clusters_)
    # print('noise count->', n_noise)
    # # print('Homogeneity-> %0.3f' % metrics.homogeneity_score(np_arr[:,1], labels))   # 동질성
    # # print("Completeness: %0.3f" % metrics.completeness_score(np_arr[:,1], labels))  # 완전성


    # cons_utm = np.empty(shape=(0,2)) #콘의 중심좌표르르 계산하기위한 초기화
    # for i in range(n_clusters_):
    #     points_of_cluster = np_arr[labels==i,:]
    #     centroid_of_cluster = np.mean(points_of_cluster, axis=0)#행방향으로 평균값을 계산
    #     cons_utm = np.vstack((cons_utm, centroid_of_cluster))
        
    # print("cluster_time : ", time.time() - c_time)

  
    # if show_plt == True:
    #     # 레이블 반복을 set으로 변환후 제거
    #     unique_labels = set(labels)

    #     # 클러스터들의 색상 부여
    #     colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    #     # DB스캔 Visualization
    #     for k, col in zip(unique_labels, colors):
    #         if k == -1:
    #             # Black used for noise.
    #             col = 'k'

    #         class_member_mask = (labels == k)

    #         # 클러스터들의 데이터를 화면에 그린다.
    #         xy = np_arr[class_member_mask & core_samples_mask]
    #         plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=u'o', alpha=0.5)

    #         # Outlier를 그린다.
    #         xy = np_arr[class_member_mask & ~core_samples_mask]
    #         plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=u'o', alpha=0.5)

    #         plt.plot(cons_utm[:,0],cons_utm[:,1], "xk" )

    #     plt.show()

    return labels