import pandas as pd
from scipy.spatial import KDTree
from .lib import mission_func as mf
import math, time
import numpy as np
import os
OFFSET = 0

class Lidar: # 라이다 클래스

    def __init__(self):

        #배달
        self.deli_flag_UTM = None
        
        #동적장애물
        self.car_stop = False

        #정적소형
        self.cone_UTM = None

        #정적대형
        self.bigcon_point = np.array([[0.0,0.0]])
        self.bigcon_sub = False
        self.points = None

        #평행주차
        self.prl_point_UTM = None
        self.prl_stop = False

        #터널
        self.tunnel_wall = None
        self.tunnel_lane = None
        self.center = None
        self.tunnel_lane_called = False
        self.tunnel_small_obj = None
        self.tunnel_small_obj_called = False
        self.tunnel_wall_called = False
        self.ceiling = None
        self.L_wall = None
        self.R_wall = None
        self.L_len = None
        self.R_len = None

        #유턴
        self.u_turn_point = None

    #배달
    def deli_point_UTM_callback(self, deli_point):
        self.deli_flag_UTM = deli_point.data

    #동적장애물
    def dynamic_callback(self,stop):
        self.car_stop = stop.data

    #정적소형
    def cone_UTM_callback(self,c_data):
        points = np.array(c_data.data)
        self.cone_UTM = points.reshape(-1, 2)

    #정적대형
    def bigcon_callback(self, obj):
        points = np.array(obj.data)
        self.bigcon_point = points.reshape(-1, 2)
        self.bigcon_sub = True
        print("-----------------대형장애물 들어옴--------------")
        

    #평행주차
    def prl_point_UTM_callback(self, prl_UTM):
        self.prl_point_UTM = prl_UTM.data
    
    def prl_stop_callback(self, prl_stop):
        self.prl_stop = prl_stop.data

    #터널
    def ternnel_wall_callback(self, wall):
        #앙 고쳐 'd'들어옴
        if len(wall.data) > 7:  # 벽 좌표를 8개이상 주고 있는지 확인.
            wall_1 = wall.data[0:4]
            wall_2 = wall.data[4:8]    
            if wall_1[1] > wall_2[1]:
                self.L_wall = wall_1
                self.R_wall = wall_2
            else:
                self.L_wall = wall_2
                self.R_wall = wall_1   
            self.L_len = math.sqrt((self.L_wall[0]-self.L_wall[2])**2 + (self.L_wall[1]-self.L_wall[3])**2)
            self.R_len = math.sqrt((self.R_wall[0]-self.R_wall[2])**2 + (self.R_wall[1]-self.R_wall[3])**2)
        
        self.tunnel_wall = wall.data
        self.tunnel_wall_called = True
    
    def ternnel_ceiling_callback(self, ceiling):
        self.ceiling = 0
        if len(ceiling.data) != 0:
            self.ceiling = ceiling.data[0]

    def ternnel_center_callback(self, center):
        if len(center.data) != 0:
            self.center = center.data

    def ternnel_lane_callback(self, lane):
        if len(lane.data) != 0:
            self.tunnel_lane = lane.data
            self.tunnel_lane_called = True
    
    def tunnel_small_callback(self, small_obj):
        points = small_obj.data
        self.tunnel_small_obj = points
        if len(points)>1:
            self.tunnel_small_obj_called = True

    #유턴
    def u_turn_UTM_callback(self,point):
        points = np.array(point.data)
        self.u_turn_point = points.reshape(-1, 2)

class Vision: # 비전 클래스

    def __init__(self):
        
        self.v_light = [0,0,0,0] #신호등 색깔여부 [빨강, 주황, 좌회전, 초록] int32mutiarry
        
        #사선주차
        self.obq_parking_points = None
        self.obq_tmp_point = None
        self.obq_parking_points = None #주차할 점 좌표 3개
        
    def v_traffic_light_callback(self,traffic):
        self.v_light = traffic.data

    #정지선
    def stopline_callback(self, line_dis):
        self.dis_stopline = line_dis.data # 공용 데이터에 dis_stopline값 저장
        
class Control: # 제어 클래스

    def __init__(self):
        self.l_path = 0
        self.c_velocity = 0 # 참고 [저속주행 : 5~6, 일반 : 20, 최대 25]
        self.c_encoder_mode = 0 # [ 1 : 풀브레이크, 2 : 엔코더 기록, 3 : 엔코더 기록 종료, 4 : 엔코더 역주행, 5 : 엔코더 사용 x]

class Local: # 로컬 클래스

    def __init__(self,mission_list,mission_dic,global_utm_txt_adress):
        
        self.global_utm_txt_adress = global_utm_txt_adress
        self.c_UTM = [-1.0,-1.0] # 현재 ERP의 UTM 좌표 저장 변수
        self.c_yaw = -1.0 # 현재 ERP의 yaw값 저장 변수

        self.map_dic = {}
        self.mission_list = mission_list
        self.mission_dic = mission_dic

        self.create_map()

        self.g_path3 = None
        self.g_yaw3 = None
        self.g_k3 = None
        self.g_kd_tree3 = None 
        self.mid_pos2 = None
        
        self.g_path2 = None
        self.g_yaw2 = None
        self.g_k2 = None
        self.g_kd_tree2 = None
        
        """ callback함수가 호출 되었는지 확인하는 변수  """
        self.lo_c_UTM_callback_called = False
        self.lo_c_yaw_callback_called = False

        """유턴"""

    """ callback function for UTM """
    def lo_c_UTM_callback(self,utm):
        self.c_UTM[0] = utm.point.x+OFFSET # 공용 데이터에 UTM의 x좌표 저장
        self.c_UTM[1] = utm.point.y+OFFSET # 공용 데이터에 UTM의 y좌표 저장
        self.lo_c_UTM_callback_called = True # 해당 callback함수가 호출됨을 알림

    """ callback function for yaw """
    def lo_c_yaw_callback(self,yaw):
        self.c_yaw = yaw.data # 공용 데이터에 yaw값 저장
        self.lo_c_yaw_callback_called = True # 해당 callback함수가 호출됨을 알림

    """ 처음 시작 미션 선택 """
    def chose_now_mission_map(self):
        dist = math.inf
        map = None
        for i in self.map_dic:
            kdtree = self.map_dic[i].g_kd_tree
            near_dist, _ = kdtree.query(self.c_UTM)
            if dist > near_dist:
                dist = near_dist
                map = self.map_dic[i]

        if map.file_name == 'big.txt':
            print('현재 가장 가까운 미션은 정적 대형 예비 입니다**')
        else:
            print('현재 가장 가까운 미션은',self.mission_dic[self.mission_list[int(map.file_name.split('.')[0])]],'미션 입니다')
                     
        while(True):
            print('-'*40)
            chose_go_ahead = int(input('그대로 갈까요?(1), 미션 선택?(2) : '))
            if chose_go_ahead == 1:
                self.g_path = map.g_path
                self.g_yaw = map.g_yaw
                self.g_k = map.g_k
                self.g_kd_tree = map.g_kd_tree
                break
            elif chose_go_ahead == 2:
                if map.file_name == 'big.txt':
                    near_mission_ind = self.mission_list.index(12)
                else:
                    near_mission_ind = int(map.file_name.split('.')[0])
                print('-'*40)
                print()
                print()
                print('*************** 이미 지나온 미션 ***************')
                for mission_ind_num in range(1,near_mission_ind):
                    print(self.mission_dic[self.mission_list[mission_ind_num]],':',mission_ind_num)
                print()
                print('############### 현재 진행중인 미션 #############')
                print(self.mission_dic[self.mission_list[near_mission_ind]],':',near_mission_ind)
                print()
                print('=============== 앞으로 진행할 미션 =============')

                for mission_num in range(near_mission_ind+1,len(self.mission_list)):
                    print(self.mission_dic[self.mission_list[mission_num]],':',mission_num)
                print()
                chose_mission_num = int(input('진행할 미션 인덱스를 입력하세요 : '))
                print('-'*40)

                if chose_mission_num < len(self.mission_list) and chose_mission_num != 0:
                    chose_mission_ind = chose_mission_num
                    if str(chose_mission_ind)+'.txt' in self.map_dic:
                        map = self.map_dic[str(chose_mission_ind)+'.txt']
                        self.g_path = map.g_path
                        self.g_yaw = map.g_yaw
                        self.g_k = map.g_k
                        self.g_kd_tree = map.g_kd_tree
                        break
                    else:
                        print('해당 미션은 현재 txt 파일이 없습니다.')
                else:
                    print('해당 미션은 진행할 미션이 아닙니다.')
            else:
                print('다시 입력해주세요')

        if map.file_name == 'big.txt' :
            return self.mission_list.index(12)
        else:
            return int(map.file_name.split('.')[0])

    """ 미션 txt 읽어오기"""
    def create_map(self):
        file_list = os.listdir(self.global_utm_txt_adress)

        for file_name in file_list:
            g_path,g_yaw,g_k = self.load_g_path_from_txt(file_name)
            self.read_map(file_name,g_path,g_yaw,g_k)

    def read_map(self, file_name, g_path, g_yaw,g_k):
        self.map_dic[file_name] = Map(file_name,g_path,g_yaw,g_k)
        
    def load_g_path_from_txt(self,f_name):
        map_data_txt = pd.read_csv(self.global_utm_txt_adress+str(f_name), sep=',', encoding='utf-8') # 파일 내용 저장 -> 2차원 리스트

        UTMmap_arr = map_data_txt.to_numpy() # 리스트를 numpy로 변환
        g_path = UTMmap_arr[:,:2]
        g_yaw = UTMmap_arr[:,2]
        g_k = UTMmap_arr[:,2]

        return g_path,g_yaw,g_k
    
    """ 미션구간을 지나갔는지 확인 """
    def is_mission_complete(self):
        dist = math.sqrt((self.g_path[-1][0]-self.c_UTM[0])**2 + (self.g_path[-1][1]-self.c_UTM[1])**2)
        #print('미션 종료까지 남은 거리 :',dist)
        return True if dist < 17 else False
    
    """ 미션 변경 """
    def change_next_mission(self,mission_num):
        change_map = self.map_dic[str(mission_num)+'.txt']
        self.g_path = change_map.g_path
        self.g_yaw = change_map.g_yaw
        self.g_k = change_map.g_k
        self.g_kd_tree = change_map.g_kd_tree

#나중에 곡률도 추가 설정 필요
class Map:
    def __init__(self,file_name,g_path,g_yaw,g_k):
        self.file_name = file_name
        self.g_path = g_path
        self.g_yaw = g_yaw
        self.g_k = g_k
        self.g_kd_tree = KDTree(self.g_path)
