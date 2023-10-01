#!/usr/bin/env python3
from . import cubic_spline_planner as csPlanner
# 미션에 따라서 라바콘을 인식하고 경로를 다시 생성함
import numpy as np
import time
import math
import copy
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from scipy import spatial
from . import dbscan, dbscan_tunnel
from sklearn.linear_model import LinearRegression
import os

CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIRECTORY = os.path.dirname(CURRENT_FILE_PATH)
A_DELI_TXT_ADDRESS = CURRENT_DIRECTORY + "/A_flag.txt"

p_cnt = 400 #3차보간 쓸 좌표 개수
PATH_TERM = 0.05
LOCAL_P = 300 #pub할 좌표 개수
SERIAL_E_STOP = -1 #E-STOP
SERIAL_NOMAL = 0 #일반주행
SERIAL_SLOW = 1 #천천히 가기
SERIAL_ACCELERATION = 2 #중속 주행
SERIAL_REVERSE = 100 #후진
SLOPE_RANGE = 40 # 40 *0.05 = 2m
#정적 대형 상수
B_OBJ_ERROR =  0.8 #0.8,2.0

# 정지선 상수
STOP_DISTANCE_PRE = 10 # 정지선 탐지시 감속할 거리.
STOP_DISTANCE = 4 # 정지선 탐지시 멈출 거리. (n 미터 앞에서 정지)
WAIT_TIME = 3.2 # 정지 대기 시간
LATE_DIS = 3 # 최소 제동 거리

STOPLINE_LIST = [[0,0],
                 [0,0],
                 [0,0],
                 [0,0],
                 [0,0],
                 [0,0],
                 [302476.8019109962,4123845.641434414, 302487.2808855878,4123854.259636674],
                 [302509.4481815001, 4123843.3522750577],
                 [0,0],
                 [302545.59076439834, 4123864.7951663504],
                 [0,0],
                 [0,0],
                 [302600.15056334063, 4123989.6214836664],
                 [0,0],
                 [0,0],
                 [302596.8684540263, 4124095.0028129937],
                 [0,0],
                 [302549.4915121738, 4124127.9582791706],
                 [0,0],
                 [302576.63600091264, 4124003.5082507697],
                 [0,0],
                 [302554.1304396321, 4123895.0142495274],
                 [0,0],
                 [302533.6910721872, 4123853.6176522253],
                 [0,0],
                 [0,0],
                 [0,0]]

TUCK = [302567.6975462873,4123753.974137217]

# 평행주차
num_points = 10 # 평행주차 점 몇개찍을지.
num_new_points = 300  # new_point 갯수 / 다은_path
ds = 0.5
STOP_TIME = 2

# 터널
FIND_NUM = 4 # head 경로 몇개 찍을지.
RATIO = 1.9 # n : 1 내분점 생성. (n값)
CAR_POINT = [0,0] # 상대좌표 상의 차량 좌표
CAR_TAIL = [[-2,0],[-1,0],[-0.5,0]] # 차량 후방 꼬리 좌표.
WALL_OFFSET = 1.1  # 터널 장애물 오프셋할 거리 
WALL_OFFSET2 = 1.9
KKK = 5.0
MM = 8
car_vec = [1,0]


""" 미션 별 글로벌패스 3차보간 """
def generate_target_course(path_raw, ds = PATH_TERM): # 경로를 일정 간격으로 자르는 함수
    """parameter
    path_raw : [[x,y],[x,y],[x,y],---],(numpy)
    ds : 생성한 경로의 점간격,(float)
    """
    x = path_raw[:, 0] # numpy배열에서 2차원 리스트의 0번 인덱스 모두 가져오기 -> [x1,x2,x3,---]
    y = path_raw[:, 1] # numpy배열에서 2차원 리스트의 1번 인덱스 모두 가져오기 -> [y1,y2,y3,---]
    csp = csPlanner.Spline2D(x, y) # cubic spline으로 3차보간
    s = np.arange(0, csp.s[-1], ds) # (csp.s[-1] = 경로의 길이, ds = 경로 내의 점 간격) -> s = [0,ds,ds*2,ds*3,---]

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    path = np.vstack((rx,ry)).T # x리스트, y리스트 vstack으로 붙은 후 전치
    """return
    path : [[x,y],[x,y],[x,y],---],(numpy)
    np.array(ryaw) : [yaw,yaw,yaw,---],(numpy)
    np.array(rk) : [k,k,k,---],(numpy)
    """
    return path, np.array(ryaw), np.array(rk) 
    
def generate_target_course_flat(path_raw, ds = 0.1): # 경로를 일정 간격으로 자르는
    """parameter
    path_raw : [x,y,x,y,x,y,---],(list)
    ds : 생성한 경로의 점간격,(float)
    """

    x = path_raw[0::2]
    y = path_raw[1::2]
    csp = csPlanner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))
    path = np.vstack((rx,ry)).T

    """return
    path : [[x,y],[x,y],[x,y],---],(numpy)
    ryaw : [yaw,yaw,yaw,---],(list)
    rk : [k,k,k,---],(list)
    """
    return path, ryaw, rk

def create_local_path(car_curr_odom,global_path,global_yaw,global_k,kd_tree): #global_path에서 local_path 추출
    """parameter
    car_curr_odom : [x,y](list)
    global_path : [[x,y],[x,y],[x,y],--- ],(numpy 배열)
    global_yaw : [yaw,yaw,yaw,---],(numpy 배열)
    global_k : [k,k,k,---],(numpy 배열)
    """
    tree_val = kd_tree.query(car_curr_odom) # car_curr_odom과 가장 가까운 global_path 점 찾기, 0 : dist, 1 : ind
    d_away = LOCAL_P # 경로의 길이 ind 300개 <-> 점간격 0.05일 때 15미터
    if len(global_path) - tree_val[1] < d_away: # global 끝지점 직전 local_path index 문제 방지
        d_away = len(global_path) - tree_val[1]
    local_path = global_path[tree_val[1]:tree_val[1]+d_away,:]
    local_yaw = global_yaw[tree_val[1]:tree_val[1]+d_away]
    local_k = global_k[tree_val[1]:tree_val[1]+d_away]
    """return
    local_path.flatten().tolist() : [x,y,x,y,x,y,---],(list)
    local_yaw.tolist() : [yaw,yaw,yaw,---],(list)
    local_k.tolist() : [k,k,k,---],(list)
    """
    return local_path.flatten().tolist() ,local_yaw.tolist(), local_k.tolist()

#배달 받기
def delivery_ready(self,car_utm,f_flag_utm):
    dist_f = abs(np.hypot(self.delivery_A_stop_point[0]-car_utm[0],self.delivery_A_stop_point[1]-car_utm[1]))
    # utm 플래그 인덱스에 저장
    if f_flag_utm is not None:
        if len(f_flag_utm) != 0:
            for i in range(len(f_flag_utm)):
                self.A_flag_index.append(f_flag_utm[2])
            self.call_flag_utm_time = time.time()
            self.deli_utm_callback = True
            self.lidar.deli_flag_UTM = None
            
    #마지막으로 들어온 표지판 데이터에서 5초이상 데이터가 안들어올때 (# if len(self.A_flag_index) >= 50: 데이터 50개 이상일때)
    if self.deli_utm_callback == True:
        print(time.time() - self.call_flag_utm_time)
    if self.deli_utm_callback == True and ((time.time() - self.call_flag_utm_time) > 5): 
        flag_dic = {}
        for A_flag in self.A_flag_index:
            if A_flag in flag_dic:
                flag_dic[A_flag] += 1
            else:
                flag_dic[A_flag] = 1

        # 제일 많은 플래그 비교
        max_count = 0
        for A_flag, count in flag_dic.items():
            if count > max_count:
                max_count = count
                if  A_flag == 1:
                    self.Deli_flag = ["A1", str(A_flag)]
                elif  A_flag == 2:
                    self.Deli_flag = ["A2", str(A_flag)]
                elif  A_flag == 3:
                    self.Deli_flag = ["A3", str(A_flag)]

        # 제일 많은 플래그를 플래그 텍스트에 저장하기
        if self.Deli_flag is not None:

            # 텍스트를 쓰기 모드로 열고 플래그 내용을 저장
            with open(A_DELI_TXT_ADDRESS,'w',encoding='utf-8') as file:
                file.write(str(self.Deli_flag[1]))
            # 사용된 변수 초기화 ( 계속해서 텍스트를 쓰지 않기 위해 )
            self.deli_utm_callback = False

    if dist_f <= 1.1 and self.fusion_state.data == 1:      #표지판과의 거리 3m 이내되면 STOP
        if self.wait == False:
            # 2m 안쪽에서 정지
            self.start_t = time.time()
            self.wait = True                          #wait 상태여부 확인
            self.c_serial_mode.data = SERIAL_E_STOP   #control_fullbreak 모드
    else:
        self.c_serial_mode.data = SERIAL_NOMAL


    if (self.wait == True) and ((time.time() - self.start_t) > 8):
        self.wait = False
        self.c_serial_mode.data = SERIAL_NOMAL
        self.fusion_state.data = 0                   #배달 준비완료
    
    if self.DBUG == True:
        print("-"*20)
        print(" A 표지판의 번호: ", self.Deli_flag)
        print(" 표지판까지의 거리 : ", dist_f)
        print(" 기다렸냐?: ", self.wait)
        print(" 인지 퓨전 상태: ", bool(self.fusion_state.data))
        print(" Control switch: ", self.c_serial_mode.data)

def fusion_deli_B(self, car_utm):
    dist_f = 0
    # 시작하면 한번만 맵을 생성하고 미션 포인트를 결정함
    if not self.deli_B_start:
        with open(A_DELI_TXT_ADDRESS,'r',encoding='utf-8') as file:
            self.Deli_flag  = float(file.read())

        #A flag -> B flag 매치
        if not self.Deli_flag == 0:
            self.B_flag = self.Deli_flag + 3
        else:
            pass
        self.fusion_state.data = 1
        self.deli_B_start = True
        self.c_serial_mode.data = SERIAL_NOMAL

    if not self.lidar.deli_flag_UTM == None and self.fusion_state.data == 1:                  # flag 정보 들어오면 시작
        print(" 인지가 표지판 줌")

        for i in range(len(self.lidar.deli_flag_UTM)):             # 들어온 Flag 정보중에 target flag 정보가 있는지

            if self.lidar.deli_flag_UTM[i] == self.B_flag:      # 있다면 맵의 로컬패스를 변경하고 제일 가까운 좌표를 선정

                self.path_flag = self.local.g_kd_tree.query((self.lidar.deli_flag_UTM[i-2],self.lidar.deli_flag_UTM[i-1]))   # 가까운 좌표 찾기
                
                self.taget_flag = True                                             # taget flag가 있음
                break             # 현재 정보 초기화
            else:
                pass                
    else:
        print(" 인지가 표지판 안줌")
    
    if self.taget_flag and self.fusion_state.data == 1:              # taget flag가 있을때 그때 제일 가까운 좌표와 현재 차량의 거리 계산 
        dist_f = abs(np.hypot((self.local.g_path[self.path_flag[1],0]-car_utm[0]),(self.local.g_path[self.path_flag[1],1]-car_utm[1])))
        if dist_f < 0.6:
            if self.wait == False:
                # 0.5m 안쪽에서 정지
                    self.start_t = time.time()
                    self.wait = True                          #wait 상태여부 확인
                    self.c_serial_mode.data = SERIAL_E_STOP               #control_fullbreak 모드
                    self.DBUG = True
        else:
            self.DBUG = False    
    
        if (self.wait == True) and ((time.time() - self.start_t) > 8.0):
            self.taget_flag = False
            self.wait = False
            self.c_serial_mode.data = SERIAL_ACCELERATION      # 감속 주행
            self.fusion_state.data = 0                  # 배달 B완료
        else:
            if not self.map_change:
                goat_map = self.local.map_dic[str(self.mission_ind+1)+'.txt']
                deli_B_g_path = goat_map.g_path
                deli_B_g_yaw = goat_map.g_yaw
                deli_B_g_k = goat_map.g_k
                deli_B_g_kd_tree = goat_map.g_kd_tree
                
                flag_xy = self.local.g_path[self.path_flag[1]]
                near_g_path_ind = self.local.g_kd_tree.query(flag_xy)[1] + 50
                near_goat_path_ind = deli_B_g_kd_tree.query(flag_xy)[1] +200

                before_g_path = self.local.g_path[near_g_path_ind-10:near_g_path_ind,:]
                before2_g_path = deli_B_g_path[near_goat_path_ind:near_goat_path_ind+10,:]

                before_path = np.vstack((before_g_path,before2_g_path))
                d_path,d_yaw,d_k = generate_target_course(before_path)

                tmp1_g_path = self.local.g_path[:near_g_path_ind-10,:]
                tmp2_g_path = deli_B_g_path[near_goat_path_ind+10:,:]

                tmp1_g_yaw = self.local.g_yaw[:near_g_path_ind-10]
                tmp2_g_yaw = deli_B_g_yaw[near_goat_path_ind+10:]

                tmp1_g_k = self.local.g_k[:near_g_path_ind-10]
                tmp2_g_k = deli_B_g_k[near_goat_path_ind+10:]
                
                self.local.g_path = np.vstack((tmp1_g_path,d_path,tmp2_g_path))
                self.local.g_yaw = np.concatenate((tmp1_g_yaw,d_yaw,tmp2_g_yaw))
                self.local.g_k = np.concatenate((tmp1_g_k,d_k,tmp2_g_k))
                self.local.g_kd_tree = KDTree(self.local.g_path)
                self.map_change = True
        
    if self.DBUG == True:
            print('-'*20)
            print(" B 표지판의 번호: ", "B"+str(self.B_flag-3))
            print(" 정지할 표지판까지의 거리 : ", dist_f)
            print(" 맵 변경? :", self.map_change)
            print(" Wait 상태: ", self.wait)
            print(" 인지 퓨전 상태: ", bool(self.fusion_state.data))
            print(" Control switch: ", self.c_serial_mode.data)

'''평행주차'''
def vector_2_radian(vector):  # 벡터를 라디안으로 바꿔주는 함수.
    angle_radians = np.arctan2(vector[1], vector[0])
    return angle_radians

def points_on_circle_arc(radius, center, start_angle, end_angle, num_points): # 원방에서 일정 범위만 잘라주는 함수.
    theta = np.linspace(start_angle, end_angle, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack((x, y))

def create_daeun_path(c_UTM,g_path,st_point,st_vec,g_kd_tree):
    distance, nearest_index = g_kd_tree.query(c_UTM)
    pre_points = g_path[max(0,nearest_index-3) : nearest_index+1]
    points = np.array([st_point + i * ds * st_vec for i in range(num_new_points)])

    # Concatenate the arrays along the first axis
    path = np.concatenate((pre_points, points))
    # print(path)
    return path

def create_out_path(global_path, prl_start_point, parking_spot, esc_D):
    new_tree = KDTree(global_path)
    distance, out_nearest_index = new_tree.query(prl_start_point)
    pre_points = global_path[out_nearest_index - int(esc_D / 0.05) : out_nearest_index + (-int(esc_D / 0.05) + 300)]
    out_path = np.vstack((parking_spot,pre_points))
    return out_path, out_nearest_index

def draw_prl_space(prl_cen, _vec, l_vec):
    ld = prl_cen + 1.5 * _vec - 2.5 * l_vec
    rd = prl_cen - 1.5 * _vec - 2.5 * l_vec
    lu = prl_cen + 1.5 * _vec + 2.5 * l_vec
    ru = prl_cen - 1.5 * _vec + 2.5 * l_vec
    prl_space = np.vstack((ld,rd,lu,ru))
    return prl_space

def create_yongseung_path(x1, y1, g_path, prl_R1, prl_R2, offset_D, offset_short, offset_gap, for_stop_D):
    if x1 is not None:
    
        # 주차 가능한 점의 좌표
        parking_center_point = np.array([x1, y1])
        
        # 가장 가까운 파란 점 찾기
        closest_point_index = np.argmin(np.linalg.norm(g_path - parking_center_point, axis=1))
        lat_point = g_path[closest_point_index]
        
        # 방향 단위 벡터 계산 (빨간 점에서 파란 점으로 향하도록)
        direction_vector = lat_point - parking_center_point
        normalized_direction_vector = direction_vector / np.linalg.norm(direction_vector) # 이것이 단위벡터
        offset_vector = np.dot(np.array([[0, -1], [1, 0]]), normalized_direction_vector)        
        parking_offset_point = parking_center_point + (offset_D) * offset_vector  - (offset_short) * direction_vector # 오프셋 점찍음

        # 1번 점.
        point_1 = parking_offset_point + (prl_R1) * normalized_direction_vector # 
        point_f = parking_offset_point + ((prl_R1 + prl_R2) / 2) * normalized_direction_vector # 
        
        # 벡터 변환 후 ST점.
        prl_vector = np.dot(np.array([[0, 1], [-1, 0]]), normalized_direction_vector)
        prl_rad = vector_2_radian(prl_vector)
        prl_st_point = point_f + (math.sqrt(3)*(prl_R1 + prl_R2)/2) * prl_vector

        # 2번 점.
        real_st_point = prl_st_point + offset_gap * normalized_direction_vector
        point_2 = real_st_point - (prl_R2) * normalized_direction_vector 

        # 1,2번 벡터 구하기.
        vec_12 = point_2 - point_1
        vec_21 = point_1 - point_2

        # vec_2_rad 변환.
        rad_12 = vector_2_radian(vec_12)
        rad_21 = vector_2_radian(vec_21)

        #원의 방정식 구하기 전에 각도 범위 구하기.
        c1_st_angle = rad_12 - np.pi/3 
        c1_fin_angle = rad_12

        c2_st_angle = rad_21
        c2_fin_angle = rad_21 - np.pi/3

        #원방 만들기
        prl_path_2 = points_on_circle_arc(prl_R2, point_2, c2_st_angle, c2_fin_angle, num_points)[::-6]
        prl_path_1 = points_on_circle_arc(prl_R1, point_1, c1_st_angle, c1_fin_angle, num_points+1)[:-1, :][::-1]
    
        # 정지 이후 예비 경로 만들기
        future_point = parking_offset_point + (for_stop_D) * offset_vector
        
        prl_path_final = np.vstack((prl_path_2,prl_path_1,future_point))
        prl_space = draw_prl_space(parking_center_point, normalized_direction_vector, prl_vector)

    return prl_path_final, real_st_point, prl_rad, prl_vector, parking_offset_point, prl_space

def prl_mission_func(self,local,lidar,s_m):
    s_m.data = SERIAL_SLOW

    if lidar.prl_stop: # if self.prl_stop 으로 변경.
        s_m.data = SERIAL_E_STOP
        print('주차공간 확인, 일단정지')

        # 추가로 다은이 코드도 해야함.
        if lidar.prl_point_UTM is not None and not self.ready_path_1:  
            s_m.data = SERIAL_E_STOP
            print('다은path 생성 중')
            self.prl_park_path, self.prl_st_point, self.prl_st_rad, self.prl_st_vec, self.prl_fin_point, self.prl_space = create_yongseung_path(lidar.prl_point_UTM[0],lidar.prl_point_UTM[1],local.g_path,self.prl_R1,self.prl_R2,self.offset_D,self.offset_short,self.offset_gap,self.for_stop_D)
            daeun_path = create_daeun_path(local.c_UTM,local.g_path,self.prl_st_point,self.prl_st_vec,local.g_kd_tree)
            print(daeun_path)
            self.prl_ready_path, self.prl_ready_yaw, self.prl_ready_k = generate_target_course(daeun_path)
            self.prl_ready_tree = KDTree(self.prl_ready_path)

            if not self.time_check:
                print('타임 췤')
                self.prl_time = time.time()
                self.time_check = True

            elif (time.time() - self.prl_time) > STOP_TIME: 
                self.ready_path_1 = True    
                self.ready_path_3 = True  
                self.time_check = False

        elif not self.ready_path_2:                 
            self.create_mission_path = True

            if self.ready_path_3:      
                print('주차 시작점으로 이동 중')
                self.l_path.data,self.l_yaw.data,self.l_k.data = create_local_path([local.c_UTM[0],local.c_UTM[1]],self.prl_ready_path,self.prl_ready_yaw,self.prl_ready_k,self.prl_ready_tree)
                s_m.data = SERIAL_NOMAL
                dis = math.sqrt((self.l_path.data[0] - self.prl_st_point[0])**2 + (self.l_path.data[1] - self.prl_st_point[1])**2)
                if  dis <= 0.1:
                    print('주차시작점 도착.')
                    self.finish_path_1 = True
                    self.ready_path_3 = False

            elif self.finish_path_1:
                print('용승path 생성 중...')
                s_m.data = SERIAL_E_STOP
                self.prl_path, self.prl_yaw, self.prl_k = generate_target_course(self.prl_park_path)
                self.prl_tree = KDTree(self.prl_path)

                if not self.time_check:
                    print('타임 췤')
                    self.prl_time = time.time()
                    self.time_check = True
                elif (time.time() - self.prl_time) > STOP_TIME: 
                    self.ready_path_2 = True
                    self.finish_path_1 = False
                    self.ready_path_4 = True
                    self.time_check = False

        elif self.ready_path_4:

            if not self.finish_path_2:
                print('주차 중...')
                self.l_path.data,self.l_yaw.data,self.l_k.data = create_local_path([local.c_UTM[0],local.c_UTM[1]],self.prl_path,self.prl_yaw,self.prl_k,self.prl_tree)
                print(self.l_path.data)
                self.create_mission_path = True
                s_m.data = SERIAL_REVERSE
                dis = math.sqrt((self.l_path.data[0] - self.prl_fin_point[0])**2 + (self.l_path.data[1] - self.prl_fin_point[1])**2)
                if dis <= 2.9:
                    self.finish_path_2 = True

            elif self.finish_path_2:
                print('주차 완료. 탈출경로 생성 중...')
                s_m.data = SERIAL_E_STOP
                # escape_path = create_out_path(local.g_path, self.prl_st_point, self.prl_fin_point, self.esc_D)
                
                # ------------------------ 아웃패스 수정 중 -------------------------------------
                map2 = local.map_dic[str(self.mission_ind+1)+'.txt']
                map2_g_path = map2.g_path
                escape_path, near_ind = create_out_path(map2_g_path, self.prl_st_point, self.prl_fin_point, self.esc_D)
                # ------------------------ 아웃패스 수정 중 -------------------------------------
                self.out_nearest_index = map2_g_path[near_ind]
                self.esc_path, self.esc_yaw, self.esc_k = generate_target_course(escape_path)
                self.esc_tree = KDTree(escape_path)

                if not self.time_check:
                    print('타임 췤')
                    self.prl_time = time.time()
                    self.time_check = True

                elif (time.time() - self.prl_time) > STOP_TIME:
                    self.create_mission_path = True
                    self.esc_path_ready = True
                    self.finish_path_2 = False
                    self.ready_path_4 = False
                    self.time_check = False

        elif self.esc_path_ready:
            print('탈출 경로 다만듬')
            if not self.finish_esc_path:
                print('탈출로컬생성')
                self.l_path.data,self.l_yaw.data,self.l_k.data = create_local_path([local.c_UTM[0],local.c_UTM[1]],self.esc_path,self.esc_yaw, self.esc_k,self.esc_tree)
                dis = math.sqrt((self.l_path.data[0] - self.out_nearest_index[0])**2 + (self.l_path.data[1] - self.out_nearest_index[1])**2)
                self.create_mission_path = True
                s_m.data = SERIAL_SLOW
                print(dis)
                if dis < 0.5:
                    print('맵변경'*100)
                    s_m.data = SERIAL_NOMAL
                    self.lidar.prl_stop = False
                    self.create_mission_path = False
                    self.mission_ind += 1
                    self.local.change_next_mission(self.mission_ind)
                    
''' 신호등 코드 '''
def stopline_left(self, c_UTM, tf_light, s_m):
    line_dis = math.sqrt((c_UTM[0] - STOPLINE_LIST[self.mission_ind][0])**2 + (c_UTM[1] - STOPLINE_LIST[self.mission_ind][1])**2)
    print("-"*20)
    print(' 정지선까지 거리 :', line_dis)
    print(' 현재 신호는 ?? :', tf_light)
    if not self.stop_ing:
        if line_dis < LATE_DIS:
            self.late_2_stop = True
    if not self.late_2_stop:
        if line_dis < STOP_DISTANCE:
            if tf_light[2] == 1:
                s_m.data = SERIAL_NOMAL
                self.stop_ing = False
            else:
                s_m.data = SERIAL_E_STOP
                self.stop_ing = True
        elif STOP_DISTANCE <= line_dis < STOP_DISTANCE_PRE:
            if tf_light[2] == 1:
                s_m.data = SERIAL_NOMAL
            else:
                s_m.data = SERIAL_SLOW
    else:
        s_m.data = SERIAL_NOMAL

def stopline_straight(self, c_UTM, tf_light, s_m):
    line_dis = math.sqrt((c_UTM[0] - STOPLINE_LIST[self.mission_ind][0])**2 + (c_UTM[1] - STOPLINE_LIST[self.mission_ind][1])**2)
    print("-"*20)
    print(' 정지선까지 거리 :', line_dis)
    print(' 현재 신호는 ?? :', tf_light)
    if not self.stop_ing:
        if line_dis < LATE_DIS:
            self.late_2_stop = True
    if not self.late_2_stop:
        if line_dis < STOP_DISTANCE:
            if tf_light[3] == 1:
                s_m.data = SERIAL_NOMAL
                self.stop_ing = False
            else:
                s_m.data = SERIAL_E_STOP
                self.stop_ing = True
        elif STOP_DISTANCE <= line_dis < STOP_DISTANCE_PRE:
            if tf_light[3] == 1:
                s_m.data = SERIAL_NOMAL
        else:
            s_m.data = SERIAL_SLOW
    else:
        s_m.data = SERIAL_NOMAL
        
'''비신호 정지선'''
def stopline_no_light(self, c_UTM, tf_light, s_m):
    if len(STOPLINE_LIST[self.mission_ind]) < 3:
        line_dis = math.sqrt((c_UTM[0] - STOPLINE_LIST[self.mission_ind][0])**2 + (c_UTM[1] - STOPLINE_LIST[self.mission_ind][1])**2)
        print("-"*20)
        print(' 정지선까지 거리 :', line_dis)
        if not self.stop_ing:
            if line_dis < STOP_DISTANCE:
                print(" 정지중 ...")
                s_m.data = SERIAL_E_STOP
                if not self.time_check:
                    print(' 타임 췤')
                    self.tf_time = time.time()
                    self.time_check = True
                    self.stop_ing = False
                elif (time.time() - self.tf_time) > 3:
                    s_m.data = SERIAL_NOMAL
                    self.stop_ing = True
                    self.time_check = False
            elif STOP_DISTANCE <= line_dis < STOP_DISTANCE_PRE:
                s_m.data = SERIAL_SLOW
                print(" 감속중 ...")
            else:
                s_m.data = SERIAL_NOMAL
                print(" 주행중 ...")
        else:
            s_m.data = SERIAL_NOMAL
    elif 3 < len(STOPLINE_LIST[self.mission_ind]):
        line_dis = math.sqrt((c_UTM[0] - STOPLINE_LIST[self.mission_ind][0])**2 + (c_UTM[1] - STOPLINE_LIST[self.mission_ind][1])**2)
        line_dis_2 = math.sqrt((c_UTM[0] - STOPLINE_LIST[self.mission_ind][2])**2 + (c_UTM[1] - STOPLINE_LIST[self.mission_ind][3])**2)
        print('-'*20)
        print(' 1번 정지선까지 거리 :', line_dis)
        print(' 2번 정지선까지 거리 :', line_dis_2)
        if not self.stop_ing:
            if STOP_DISTANCE <= line_dis < STOP_DISTANCE_PRE:
                s_m.data = SERIAL_SLOW
            elif line_dis < STOP_DISTANCE:
                s_m.data = SERIAL_E_STOP
                print('1번 정지 중...')
                if not self.time_check:
                    self.tf_time = time.time()
                    self.time_check = True
                    self.stop_ing = False
                elif (time.time() - self.tf_time) > 3:
                    s_m.data = SERIAL_SLOW
                    self.stop_ing = True
                    self.time_check = False

        if not self.stop_ing_2 and self.stop_ing:
            if STOP_DISTANCE <= line_dis_2 < STOP_DISTANCE_PRE:
                s_m.data = SERIAL_SLOW
            elif line_dis_2 < STOP_DISTANCE:
                s_m.data = SERIAL_E_STOP
                print('2번 정지 중...')
                if not self.time_check:
                    self.tf_time = time.time()
                    self.time_check = True
                    self.stop_ing_2 = False
                elif (time.time() - self.tf_time) > 3:
                    s_m.data = SERIAL_NOMAL
                    self.stop_ing_2 = True
                    self.time_check = False

#정적 소형
def small_object_fuc(self,local,lidar,s_m):
    
    global_path = local
    
    #회피점 offset_D
    move_first = 2.0
    move_first2 = 2.0
    move_second = 2.0

    #첫번째 보간점 선정
    f_csp1 = 180
    f_csp2 = 60

    #두번째 보간점 선정
    s_csp1 = 150
    s_csp2 = 180

    if self.tae_path is None:
        self.tae_path = copy.deepcopy(global_path)

    if self.standard_path is None:
        self.standard_path = copy.deepcopy(global_path) # 라바콘과 Global Path 계산 시 변형되지 않은 Global Path로 계산하기 위해 복제해둔다.

    if lidar.cone_UTM is not None and self.Tmap == False : # Tmap은 Local Path 생성 과정을 한 번만 하기 위한 스위치이다.
        cone = lidar.cone_UTM
        min_dis, smin_dis = math.inf, math.inf
        fcp = None
        scp = None

        valid_cone_array = []
        vaild_cone_D_array = []

        for i, p in enumerate(cone):
            tree_val = self.tae_path.g_kd_tree.query((p[0], p[1])) # e2f.get_lateral_dist(standard_path.g_p[:,0], standard_path.g_p[:,1], cone[i][0], cone[i][1])
            dis = tree_val[0]
            print("distance :", dis)
            if abs(dis) > 1.8 :  # 글로벌 패스에서 폭 범위를 설정히여 장애물을 걸러낸다.
                print("차선 밖 장애물이 걸리짐")
                continue
            else:
                relative_cone_d = np.hypot(p[0] - local.c_UTM[0], p[1] - local.c_UTM[1])
                valid_cone_array.append(p)
                vaild_cone_D_array.append(relative_cone_d)
        
        # print("con", valid_cone_array, "dis", dist_cone_array)

        if len(vaild_cone_D_array) > 0:       
            for i in range(len(vaild_cone_D_array)):
                first_distance = min(vaild_cone_D_array)
                first_index = vaild_cone_D_array.index(first_distance)
                fcp = valid_cone_array[first_index]
            valid_cone_array.pop(first_index)
            vaild_cone_D_array.pop(first_index)

            for j in range(len(vaild_cone_D_array)):
                second_distance = min(vaild_cone_D_array)
                second_index = vaild_cone_D_array.index(second_distance)
                scp = valid_cone_array[second_index]


        if fcp is not None :
            fc = fcp
            
            if scp is not None :
                sc = scp
            
            else : 
                pass
        
        print('first', fcp)
        print('second', scp)
        
        if fcp is not None :
            small_d = np.hypot(fc[0] - local.c_UTM[0], fc[1] - local.c_UTM[1])

            if small_d >= 6.0  and small_d <= 9.0 and self.tmap == False:
                fst_val= self.standard_path.g_kd_tree.query((fc[0], fc[1]))
                fst_dist = fst_val[0]
                fst_index = fst_val[1]+1

                f_on_path = self.standard_path.g_path[fst_index, :]

                # forward_vector[0] = (fc[1] - f_on_path[1]) / abs(fst_dist)
                # forward_vector[1] = -(fc[0] - f_on_path[0]) / abs(fst_dist)
                # compar_vector[0] = (local.c_UTM[0]-f_on_path[0])
                # compar_vector[1] = (local.c_UTM[1]-f_on_path[1])

                # if compar_vector[0]*forward_vector[0] + compar_vector[1]*forward_vector[1] < 0 :
                #     forward_vector[0] = - forward_vector[0]
                #     forward_vector[1] = - forward_vector[1]
                # else : 
                #     pass

                self.f_target_p[0] = fc[0] - (fc[0]-f_on_path[0]) / abs(fst_dist) * move_first # + forward_vector[0] * pull_forward # 장매물 좌표에서 글로벌 패스와 최소 거리를 갖는 인덱스 점 방향으로 2.2m 떨어진 지점에 점을 찍겠다. 
                self.f_target_p[1] = fc[1] - (fc[1]-f_on_path[1]) / abs(fst_dist) * move_first # + forward_vector[1] * pull_forward # forward_vector를 사용하여 회피점을 숫자(m)만큼 앞에 찍도록 한다.

                # 특정 값을 설정하여 글로벌 패스를 자르고 그 부분에 대해서만 generate_target_course를 해주어 계산량을 줄인다.
                tmp_path1 = global_path.g_path[fst_index-f_csp1,  :]
                tmp_path2 = self.f_target_p
                tmp_path3 = global_path.g_path[fst_index+ f_csp2, :]

                # stacking
                small_path = np.vstack((tmp_path1, tmp_path2))
                small_path = np.vstack((small_path, tmp_path3))
                
                self.tmap = True # 첫 번쨰 꼬깔이 인지 되었다. 두 번째 꼬깔을 보기 위한 준비가 되었다.
                t_path, t_yaw, t_k = generate_target_course(small_path, ds=0.05)

                local.g_path = np.concatenate((global_path.g_path[:fst_index - f_csp1, :], t_path, global_path.g_path[fst_index + f_csp2:, :]),axis=0) # vstack으로는 오류가 있어서 변경되는 부분을 concatenate를 활용하여 기존의 path와 붙여주었다(list 수 오류?)
                local.g_yaw = np.concatenate((global_path.g_yaw[:fst_index - f_csp1],np.array(t_yaw),global_path.g_yaw[fst_index + f_csp2:]),axis = 0)
                local.g_k = np.concatenate((global_path.g_k[:fst_index - f_csp1],np.array(t_k),global_path.g_k[fst_index + f_csp2:]),axis=0)               
                local.g_kd_tree = KDTree(local.g_path)
                print("첫번째 경로 변경 완료")

            elif small_d < 6.0 and self.tmap == True and scp is not None :

                fst_val= self.standard_path.g_kd_tree.query((fc[0], fc[1]))
                fst_dist = fst_val[0]
                fst_index = fst_val[1]

                sc_val= self.standard_path.g_kd_tree.query((sc[0], sc[1]))
                sed_dist = sc_val[0]
                sed_index = sc_val[1]

                f_on_path = self.standard_path.g_path[fst_index, :]
                s_on_path = self.standard_path.g_path[sed_index, :]

                # forward_vector[0] = (fc[1] - f_on_path[1]) / abs(fst_dist)
                # forward_vector[1] = -(fc[0] - f_on_path[0]) / abs(fst_dist)
                # compar_vector[0] = (local.c_UTM[0]-f_on_path[0])
                # compar_vector[1] = (local.c_UTM[1]-f_on_path[1])

                # if compar_vector[0]*forward_vector[0] + compar_vector[1]*forward_vector[1] < 0 :
                #     forward_vector[0] = - forward_vector[0]
                #     forward_vector[1] = - forward_vector[1]
                # else : 
                #     pass
                
                
                self.f_target_p[0] = fc[0] - (fc[0]-f_on_path[0]) / abs(fst_dist) * move_first2 # + forward_vector[0] * pull_forward
                self.f_target_p[1] = fc[1] - (fc[1]-f_on_path[1]) / abs(fst_dist) * move_first2 # + forward_vector[1] * pull_forward
                self.s_target_p[0] = sc[0] - (sc[0]-s_on_path[0]) / abs(sed_dist) * move_second # + forward_vector[0] * pull_forward
                self.s_target_p[1] = sc[1] - (sc[1]-s_on_path[1]) / abs(sed_dist) * move_second # + forward_vector[1] * pull_forward

                # indexing
                tmp_path1 = self.standard_path.g_path[fst_index-s_csp1,  :]
                tmp_path2 = self.f_target_p
                tmp_path3 = self.s_target_p
                tmp_path4 = self.standard_path.g_path[sed_index+s_csp2,  :]

                # stacking
                small_path = np.vstack((tmp_path1, tmp_path2))
                small_path = np.vstack((small_path, tmp_path3))
                small_path = np.vstack((small_path, tmp_path4))

                t2_path, t2_yaw, t2_k = generate_target_course(small_path, ds=0.05)
                
                local.g_path = np.concatenate((self.standard_path.g_path[:fst_index - s_csp1, :], t2_path, self.standard_path.g_path[sed_index + s_csp2:, :]))
                local.g_yaw = np.concatenate((self.standard_path.g_yaw[:fst_index - s_csp1],np.array(t2_yaw),self.standard_path.g_yaw[sed_index + s_csp2:]),axis = 0)
                local.g_k = np.concatenate((self.standard_path.g_k[:fst_index - s_csp1],np.array(t2_k),self.standard_path.g_k[sed_index + s_csp2:]),axis=0)  
                local.g_kd_tree = KDTree(local.g_path)

                self.Tmap = True # Local_path를 한 번만 만들기 위한 장치, 만들었기 때문에 True 값을 반환
                self.tmap = False
                print("최종 경로 변경 완료") 
            
    else :
        pass


def sure_ob_classification(pos1,kd_tree,g_path): #pos1 -> 끝점, kd_tree -> global_path의 kd_tree, g_path -> 해당 global_path
    dis,ind = kd_tree.query(pos1)
    
    c = g_path[ind] # pos1에서 글로벌 패스로 수직으로 내렸을때 가장 가까운 점 c!!
    c_10 = g_path[ind+2]

    v1x = pos1[0] - c[0] #2.벡터 v1 (back_p -> 현재 위치에서 가장 가까운 global_path) 
    v1y = pos1[1] - c[1]
    v1 = [v1x,v1y]

    v2x = c_10[0] - c[0]#3.벡터 v2 (back_p -> 차의 현재 위치)
    v2y = c_10[1] - c[1]
    v2 = [v2x,v2y]

    cross_product=v2[0]*v1[1]- v2[1]*v1[0]

    return cross_product


def bbangbbang_path(self,local,lidar): # g_path -> global_path.g_p,global_path.g_p2

    if local.g_path3 is None:
        local.g_path3 = copy.deepcopy(local.g_path)
        local.g_yaw3 = copy.deepcopy(local.g_yaw)
        local.g_k3 = copy.deepcopy(local.g_k)
        local.g_kd_tree3 = copy.deepcopy(local.g_kd_tree)

        big_map = self.local.map_dic['big.txt']
        local.g_path2 = big_map.g_path
        local.g_yaw2 = big_map.g_yaw
        local.g_k2 = big_map.g_k
        local.g_kd_tree2 = big_map.g_kd_tree

    if self.big_step == 0: 
        min_dis = math.inf
        s_p = []

        for s in range(0, len(lidar.bigcon_point) - 1, 2):
            p = lidar.bigcon_point[s]
            next_p = lidar.bigcon_point[s + 1]
            
            mid_p = ((p[0] + next_p[0]) / 2, (p[1] + next_p[1]) / 2)
            
            end_1 = sure_ob_classification(p, local.g_kd_tree, local.g_path)
            end_2 = sure_ob_classification(next_p, local.g_kd_tree, local.g_path)
            
            if end_1 * end_2 < 0:
                print("~~~~~~제어 추종해줘~~~")
                self.c_serial_mode.data = SERIAL_NOMAL
                mid_pos = local.g_kd_tree.query(mid_p)
                print('11111현재 글로벌 패스 안에 장애물 있다잉1111')
                print(mid_pos[1])

                if abs(mid_pos[0]) > B_OBJ_ERROR:  # 경로와 점의 거리가 B_OBJ_ERROR m가 넘으면 도로 밖
                    continue
                if abs(mid_pos[0]) < min_dis:
                    min_dis = abs(mid_pos[0])
                    local.mid_pos2 = local.g_kd_tree2.query(local.g_path[mid_pos[1]])
                    s_p = mid_p

        if len(s_p) != 0:
            self.count_big_obj += 1
        else: self.count_big_obj -= 1

        if self.count_big_obj < 0:
            self.count_big_obj = 0
        elif self.count_big_obj > 10:
            self.count_big_obj = 10

        print("확정 장애물:",self.count_big_obj)
        if self.count_big_obj > 4:
            print("첫번째 경로변경준비--------")
            f_, fst_index = local.g_kd_tree.query(s_p)
            # self.change_point = local.g_path[fst_index]
            fst_index -= 80 #기준 2m
            s_, sed_index = local.g_kd_tree2.query(local.g_path[fst_index])

            smooth_ind = 80 #간격 2m

            tmp_path1 = local.g_path[fst_index-smooth_ind-5:fst_index-smooth_ind,:] # 장애물로부터 4m 50cm 전에서 4m까지
            tmp_path2 = local.g_path2[sed_index+smooth_ind:sed_index+smooth_ind+5,:]# 
            
            before_path = np.vstack((tmp_path1,tmp_path2))

            after_path,after_yaw,after_k = generate_target_course(before_path) # csp는 정말 필요한 극소의 부분만 하는것이 좋다

            local.g_path = np.vstack((local.g_path[:fst_index-smooth_ind-5,:],after_path,local.g_path2[sed_index+smooth_ind:,:]))
            local.g_yaw = np.concatenate((local.g_yaw[:fst_index-smooth_ind-5],after_yaw,local.g_yaw2[sed_index+smooth_ind:]))
            local.g_k = np.concatenate((local.g_k[:fst_index-smooth_ind-5],after_k,local.g_k2[sed_index+smooth_ind:]))
            local.g_kd_tree = spatial.KDTree(local.g_path)

            self.big_step += 1
            self.count_big_obj = 0
            print('---------첫번째경로변경----------')

    elif self.big_step == 1:
        min_dis = math.inf
        s_p = []
        self.c_serial_mode.data = SERIAL_NOMAL
        print("제어 추종해줘~~~")

        for s in range(0,len(lidar.bigcon_point) - 1,2):
        
            p = lidar.bigcon_point[s]
            next_p = lidar.bigcon_point[s + 1]

            mid_p = ((p[0] + next_p[0]) / 2, (p[1] + next_p[1]) / 2)

            end_1 = sure_ob_classification(p, local.g_kd_tree2, local.g_path2)
            end_2 = sure_ob_classification(next_p, local.g_kd_tree2, local.g_path2)
        
            if end_1 * end_2 < 0:
                print('22222현재 글로벌 패스 안에 장애물 있다잉22222')
                mid_ppos = local.g_kd_tree2.query(mid_p)
                print("첫번째 두번째 ob 비교: ", mid_ppos[1],local.mid_pos2[1])

                if abs(mid_ppos[0]) > B_OBJ_ERROR:  # 경로와 점의 거리가 B_OBJ_ERROR m가 넘으면 도로 밖
                    continue
                if abs(mid_ppos[0]) < min_dis and mid_ppos[1] > local.mid_pos2[1]: 
                    min_dis = abs(mid_ppos[0])
                    s_p = mid_p


        if len(s_p) != 0:
            self.count_big_obj += 1
        else: self.count_big_obj -= 1

        if self.count_big_obj < 0:
            self.count_big_obj = 0

        elif self.count_big_obj > 10:
            self.count_big_obj = 10
        
        
        if self.count_big_obj > 4 :  # 한 3번 받으면 믿어도 되지 않을까?
            print('222222두번째 경로변경2222')
            f_, fst_index = local.g_kd_tree.query(s_p)
            fst_index -= 30 #기준 3m
            s_, sed_index = local.g_kd_tree3.query(local.g_path[fst_index])

            smooth_ind = 80 #간격 2m

            tmp_path1 = local.g_path[fst_index-smooth_ind-5:fst_index-smooth_ind,:]
            tmp_path2 = local.g_path3[sed_index+smooth_ind:sed_index+smooth_ind+5,:]
            
            before_path = np.vstack((tmp_path1,tmp_path2))

            after_path,after_yaw,after_k = generate_target_course(before_path)

            local.g_path = np.vstack((local.g_path[:fst_index-smooth_ind-5,:],after_path,local.g_path3[sed_index+smooth_ind:,:]))
            local.g_yaw = np.concatenate((local.g_yaw[:fst_index-smooth_ind-5],after_yaw,local.g_yaw3[sed_index+smooth_ind:]))
            local.g_k = np.concatenate((local.g_k[:fst_index-smooth_ind-5],after_k,local.g_k3[sed_index+smooth_ind:]))
            local.g_kd_tree = spatial.KDTree(local.g_path)

            self.count_big_obj = 0
            self.real_count = 0
            self.big_step += 1
            print('2222222222222222222222222222222222222222경로 바꿈 완료222222222222222222222222222222222222')

''' ================================================== 터널 주행 ================================================='''

def divide_point(x,y,ratio):
    x_R,y_R = x
    x_L,y_L = y
    x_internal = (x_L + ratio * x_R) / (1 + ratio)
    y_internal = (y_L + ratio * y_R) / (1 + ratio)
    sub = np.array([x_internal,y_internal])
    return sub

def distance_from_origin(point):
    return math.sqrt(point[0]**2 + point[1]**2)

def sort_coordinates_by_distance(coordinates):
    return sorted(coordinates, key=distance_from_origin)

def generate_equidistant_coordinates(start, end, num_points):
    if num_points <= 0:
        return []
    min_x, min_y = start
    max_x, max_y = end
    step_x = (max_x - min_x) / (num_points - 1)
    step_y = (max_y - min_y) / (num_points - 1)
    equidistant_coordinates = []
    for i in range(num_points):
        x = min_x + step_x * i
        y = min_y + step_y * i
        equidistant_coordinates.append([x, y])
    return equidistant_coordinates

def get_sorted_indices(lst):
    sorted_indices = sorted(range(len(lst)), key=lambda x: lst[x], reverse=True)
    return sorted_indices

def regression(point):
    xx = np.array([])
    yy = np.array([])
    reg_points = np.array([])
    for i in range(len(point)//2):
        x = [point[2*i]]
        y = [point[2*i + 1]]
        xx = np.append(xx,x)
        yy = np.append(yy,y)
    model = LinearRegression()
    model.fit(xx.reshape(-1,1),yy.reshape(-1,1))
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    for i in range(15):
        reg_temp = [i, slope * i + intercept]
        reg_points = np.append(reg_points,reg_temp)

    return reg_points

def wall_drive(self,lidar,s_m):
    l2D, r2D = [], []
    wall_cen_point_pre = np.array([])
    go = np.array([])
    self.first_line = None
    self.second_line = None
    self.third_line = None
    self.tunnel_path = None
    self.cen_path = None

    l_points = np.empty((0, 2))
    r_points = np.empty((0, 2))

    slope_l = (lidar.L_wall[3] - lidar.L_wall[1]) / (lidar.L_wall[2] - lidar.L_wall[0])
    slope_r = (lidar.R_wall[3] - lidar.R_wall[1]) / (lidar.R_wall[2] - lidar.R_wall[0])
    intercept_l = lidar.L_wall[1] - slope_l * lidar.L_wall[0]
    intercept_r = lidar.R_wall[1] - slope_r * lidar.R_wall[0]

    for i in range(13 * 2):
        l_temp = [i/2, slope_l * i/2 + intercept_l]
        r_temp = [i/2, slope_r * i/2 + intercept_r]
        l_points = np.append(l_points,l_temp)
        r_points = np.append(r_points,r_temp)
    self.L_wall = l_points
    self.R_wall = r_points
    change_2D = False
    if not change_2D and self.L_wall is not None and self.R_wall is not None:
        for i in range(len(self.L_wall)//2):
            l2D.append([self.L_wall[2*i],self.L_wall[2*i+1]])
            r2D.append([self.R_wall[2*i],self.R_wall[2*i+1]])
        change_2D = True
    self.L_wall_2D = np.array(l2D).reshape((-1, 2)) 
    self.R_wall_2D = np.array(r2D).reshape((-1, 2)) 
    l_tree = KDTree(self.L_wall_2D)
    r_tree = KDTree(self.R_wall_2D)
    for i in range(FIND_NUM):
        ind = FIND_NUM - i
        dis, near_index = l_tree.query(self.R_wall_2D[-ind])            
        wall_cen_each = np.array(divide_point(self.R_wall_2D[-ind], self.L_wall_2D[-ind], RATIO))
        wall_cen_point_pre = np.append(wall_cen_point_pre, wall_cen_each, axis=0)
    
    wall_cen_point = wall_cen_point_pre.reshape((-1, 2)) 
    wall_road_vec = wall_cen_point[1] - wall_cen_point[0]
    wall_road_norm_vec = wall_road_vec / np.linalg.norm(wall_road_vec)
    wall_small_offset_vec = np.dot(np.array([[0, -1], [1, 0]]), wall_road_norm_vec) # 장애물 오른쪽 방향 노멀벡터
    wall_car_vec = [1,0]
    wall_yawvec = wall_road_vec[0] * wall_car_vec[0] + wall_road_vec[1] * wall_car_vec[1]
    wall_cos_theta = wall_yawvec / (math.sqrt(wall_road_vec[0]**2 + wall_road_vec[1]**2))
    wall_car_yaw = math.acos(wall_cos_theta)
    
    cross_product_wall = wall_road_vec[0] * wall_car_vec[1] - wall_road_vec[1] * wall_car_vec[0]
    if cross_product_wall < 0:
        wall_car_yaw = -wall_car_yaw
    self.car_yaw.data = wall_car_yaw
    
    moving_point1 = []
    moving_point2 = []

    if lidar.tunnel_small_obj is not None and not self.small_ing:
        print("장애물 발견")
        if len(lidar.tunnel_small_obj) < 2:
            go = np.vstack((np.array(CAR_TAIL), wall_cen_point)) 
        elif 2 < len(lidar.tunnel_small_obj) < 4:
            car_dis = math.sqrt((lidar.tunnel_small_obj[0]-0)**2 + (lidar.tunnel_small_obj[1]-0)**2)
            print(car_dis)
            print('한개뜸')
            print('그냥장애물거리 : ',car_dis)
            obj1 = [lidar.tunnel_small_obj[0],lidar.tunnel_small_obj[1]]
            if car_dis < MM:
                dis, _ = r_tree.query(obj1)
                print(dis)
                if 2 < dis < KKK:
                    moving_point1 = obj1 + (WALL_OFFSET2) * wall_small_offset_vec
                    if len(moving_point1) != 0:
                        go = np.vstack((np.array(CAR_TAIL), moving_point1, wall_cen_point)) 
                
                elif KKK <= dis < 8:
                    moving_point1 = obj1 - (WALL_OFFSET2) * wall_small_offset_vec
                    if len(moving_point1) != 0:
                        go = np.vstack((np.array(CAR_TAIL), moving_point1, wall_cen_point)) 

                else:
                    go = np.vstack((np.array(CAR_TAIL), wall_cen_point)) 
            else:
                go = np.vstack((np.array(CAR_TAIL), wall_cen_point))

        elif 4 < len(lidar.tunnel_small_obj) :
            print('둘 다 발견')
            car_dis = math.sqrt((lidar.tunnel_small_obj[0]-0)**2 + (lidar.tunnel_small_obj[1]-0)**2)
            car_dis1 = math.sqrt((lidar.tunnel_small_obj[2]-0)**2 + (lidar.tunnel_small_obj[3]-0)**2)
            print('첫장애물 거리 : ',car_dis)
            print('두번째 장애물 거리 : ',car_dis1)
            obj1 = [lidar.tunnel_small_obj[0],lidar.tunnel_small_obj[1]]
            obj2 = [lidar.tunnel_small_obj[2],lidar.tunnel_small_obj[3]]
            if math.sqrt(obj1[0]**2 + obj1[1]**2) < math.sqrt(obj2[0]**2 + obj2[1]**2):
                fobj = obj1
                sobj = obj2
            else:
                fobj = obj2
                sobj = obj1
            
            if car_dis < MM:
                dis1, _ = r_tree.query(fobj)
                print(dis1)
                
                if 2 < dis1 < KKK:
                    moving_point1 = fobj + (WALL_OFFSET2) * wall_small_offset_vec                    
                elif KKK <= dis1 < 8:
                    moving_point1 = fobj - (WALL_OFFSET2) * wall_small_offset_vec 


            if car_dis1 < MM:
                dis2, _ = r_tree.query(sobj)
                print(dis2)
                if 2 < dis2 < KKK:
                    moving_point2 = sobj + (WALL_OFFSET) * wall_small_offset_vec
                elif KKK <= dis2 < 8:
                    moving_point2 = sobj - (WALL_OFFSET) * wall_small_offset_vec
            if len(moving_point1) != 0 and len(moving_point2) != 0:
                go = np.vstack((np.array(CAR_TAIL), moving_point1, moving_point2, wall_cen_point)) 
            elif len(moving_point1) == 0 and len(moving_point2) == 0:
                go = np.vstack((np.array(CAR_TAIL), wall_cen_point))
            elif len(moving_point1) != 0 and len(moving_point2) == 0:   
                go = np.vstack((np.array(CAR_TAIL), moving_point1, wall_cen_point)) 
            elif len(moving_point1) == 0 and len(moving_point2) != 0:  
                go = np.vstack((np.array(CAR_TAIL), moving_point2, wall_cen_point)) 

            self.tunnel_small_check = time.time() 

    elif self.mission_num == 17:
        go = np.vstack((np.array(CAR_TAIL), wall_cen_point)) 
    else:
        print('장애물 미발견')
        go = np.vstack((np.array(CAR_TAIL), wall_cen_point)) 

    if self.change_flag:
        print('동적 on')
        go = np.vstack((np.array(CAR_TAIL), wall_cen_point)) 

    self.go = go
    if len(go) > 1:
        wall_path, wall_yaw, wall_k = generate_target_course(go) 
        print(float(wall_yaw[0]))
        wall_path = wall_path[wall_path[:, 0] >= 0]
        self.wall_path = np.vstack(([-82.82,-82.82],wall_path))  
        self.l_path.data = self.wall_path.flatten().tolist()
        self.l_yaw.data = wall_yaw.tolist()
        self.l_k.data = wall_k.tolist()

def lane_drive(self,lidar,s_m):
    cen_point = np.array([])
    if lidar.tunnel_lane is not None: # 아닐때는 차선주행.
        self.L_wall = None
        self.R_wall = None
        self.wall_path = None
        self.cen_path = None
        change_2D = False
        self.tunnel_path = None
        outsider = np.empty((0, 2))
        first = np.empty((0, 2))
        second = np.empty((0, 2))
        third = np.empty((0, 2))
        self.outsider = np.empty((0, 2))
        self.first = np.empty((0, 2))
        self.second = np.empty((0, 2))
        self.third = np.empty((0, 2))
        if not change_2D and lidar.tunnel_lane is not None:
            for i in range(len(lidar.tunnel_lane)//2):
                self.tunnel_lane_2D.append([lidar.tunnel_lane[2*i],lidar.tunnel_lane[2*i+1]])
            change_2D = True
        db_lane = dbscan_tunnel.np_dbscan(np.array(self.tunnel_lane_2D))

        if max(db_lane) > 1:  # 0 :차선이 두개만 보여도 간다는 뜻 / 1 : 차선 3개일때만 가는거
            for i in range(len(db_lane)):
                if db_lane[i] == -1:
                    outsider = np.append(outsider,[self.tunnel_lane_2D[i]],axis=0)
                elif db_lane[i] == 0:
                    first = np.append(first,[self.tunnel_lane_2D[i]],axis=0)
                elif db_lane[i] == 1:
                    second = np.append(second,[self.tunnel_lane_2D[i]],axis=0)
                elif db_lane[i] == 2:
                    third = np.append(third,[self.tunnel_lane_2D[i]],axis=0)
            result = tuple(get_sorted_indices([first[0,1],second[0,1],third[0,1]]))
            order_mapping = {
                (0, 1, 2): [first, second, third],
                (0, 2, 1): [first, third, second],
                (1, 0, 2): [second, first, third],
                (1, 2, 0): [second, third, first],
                (2, 0, 1): [third, first, second],
                (2, 1, 0): [third, second, first]
            }

            self.first, self.second, self.third = order_mapping[result]
            self.first_order = np.array(sort_coordinates_by_distance(self.first)) 
            self.second_order = np.array(sort_coordinates_by_distance(self.second))
            self.third_order = np.array(sort_coordinates_by_distance(self.third))
            self.first_line = np.array(regression(self.first_order.reshape(-1, 1))).reshape((-1,2))
            self.second_line = np.array(regression(self.second_order.reshape(-1, 1))).reshape((-1,2))
            self.third_line = np.array(regression(self.third_order.reshape(-1, 1))).reshape((-1,2))
            l_tree = KDTree(self.second)
            # print(l_tree)
            for i in range(FIND_NUM):
                ind = FIND_NUM - i
                dis, near_index = l_tree.query(self.third_line[-ind])                        
                cen_point_each = np.array(divide_point(self.third_line[-ind], self.second_line[-ind], 1))
                cen_point = np.append(cen_point, cen_point_each, axis=0)
            
            cen_point = cen_point.reshape((-1, 2))  # 2개의 열을 갖도록 배열 모양을 변경
            self.cen_point = cen_point
            gogo = np.vstack((np.array(CAR_TAIL), cen_point))
            path, yaw, k = generate_target_course(gogo) 
            path = path[path[:, 0] >= 0]
            self.tunnel_path = np.vstack(([-82.82,-82.82],path))  # 키고 밑에 끄고 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            self.l_path.data = self.tunnel_path.flatten().tolist()
            self.l_yaw.data = yaw.tolist()
            self.l_k.data = k.tolist()
            road_vec = cen_point[1] - cen_point[0]
            product_yawvec = road_vec[0] * car_vec[0] + road_vec[1] * car_vec[1]
            cos_theta = product_yawvec / (math.sqrt(road_vec[0]**2 + road_vec[1]**2))
            car_yaw = math.acos(cos_theta)
            
            cross_product = road_vec[0] * car_vec[1] - road_vec[1] * car_vec[0]
            if cross_product < 0:
                car_yaw = -car_yaw
            self.car_yaw.data = car_yaw

        self.tunnel_lane_2D = []

def cen_drive(self,lidar,s_m):
    self.L_wall = None
    self.R_wall = None
    self.wall_path = None
    self.first_line = None
    self.second_line = None
    self.third_line = None
    self.tunnel_path = None
    print(np.array(lidar.center).reshape((-1, 2)))
    cen_point = np.array(lidar.center).reshape((-1, 2))
    further_vec = [cen_point[-1][0] - cen_point[-2][0],cen_point[-1][1] - cen_point[-2][1]]
    norm_further_vec = further_vec / np.linalg.norm(further_vec)
    further_point = cen_point[-1] + norm_further_vec * 10
    gogo = np.vstack((np.array(CAR_TAIL), cen_point, further_point))
    print(gogo)
    path, yaw, k = generate_target_course(gogo)
    path = path[path[:, 0] >= 0]
    self.cen_path = np.vstack(([-82.82,-82.82],path))
    self.l_path.data = self.cen_path.flatten().tolist()
    self.l_yaw.data = yaw.tolist()
    self.l_k.data = k.tolist()


def drive_turnel(self,lidar,s_m):
    a = 10
    self.create_mission_path = True
    if lidar.center is None:
        self.lane_mode = False
    else:
        self.lane_mode = True

    if lidar.L_len is not None and lidar.R_len is not None:
        print("LLLLLLLLLLLL", lidar.L_len)
        print("RRRRRRRRRRRR", lidar.R_len)
        if lidar.R_len >= 8 and lidar.L_len >= 12:
            print("벽보는 중 :", self.elapsed_time)
            if self.start_time is None:
                self.start_time = time.time()
            else:
                self.elapsed_time = time.time() - self.start_time
                if self.elapsed_time >= 2:
                    self.wall_check = True
        else:
            self.start_time = None
            self.wall_check = False
            print('@@@@@@@@@초기화초기화@@@@@@@@@')
        
    if self.wall_check:
        if lidar.ceiling is not None:
            print("천장",lidar.ceiling)
            if not self.in_tunnel and lidar.ceiling > 30:  # 벽 안으로 첫 진입 시 천장 커트라인 70
                self.wall_mode = True
                self.in_tunnel = True
            else:  # 벽 안으로 들어옴.
                if lidar.ceiling == 0:  # 천장이 없으면 N초 뒤 gps로 변경.
                    if self.gps_time is None:
                        self.gps_time = time.time()
                    else:
                        if time.time() - self.gps_time > 8:
                            self.gps_mode = True
                elif lidar.ceiling < 30 or lidar.R_len <= 8 or lidar.L_len <= 12:  # 천장이 있긴 한데 30보다 작거나, 벽길이 짧으면 벽 모드 끄기.
                    self.gps_time = None
                    if self.ceil_time is None:
                        self.ceil_time = time.time()
                    else:
                        if time.time() - self.ceil_time > 2:  # 천장이 2초이상 짧아진게 확인됬다.
                            self.wall_mode = False
                else:
                    self.wall_time = None
                    self.gps_time = None
                    self.ceil_time = None
                    self.wall_mode = True

    if self.gps_mode: # gps주행
        self.create_mission_path = False
        if not self.change_gps:
            self.mission_ind += 1
            self.change_gps = True
        print("================= gps Mode =================")
    
    elif self.wall_mode:
        print("================ 벽보고 주행중 ================")
        wall_drive(self, lidar, s_m)
    
    elif self.lane_mode:
        print("================= 중앙주행중 =================")
        cen_drive(self,lidar,s_m)

    else:
        print("================= 아무것도 없다!!!!!! ================")
        # lane_drive(self,lidar,s_m)

    if self.tunnel_small_check is not None:  # 소형에서 동적으로 플래그 넘기는 부분
        flag_time = time.time() - self.tunnel_small_check
        print("플래그 바꾸실?? :", flag_time)
        if flag_time > 5 and not self.change_flag:
            print('플래그넘기자~~~~~~~~')

            self.change_flag = True
            self.mission_ind += 1

    if lidar.car_stop:  # 동적장애물 코드
        s_m.data = SERIAL_E_STOP
        if not self.check1:
            self.time_stop = time.time()
            self.check1 = True
    elif self.check1:
        s_m.data = SERIAL_E_STOP
        print("멈추는 중임")
        if time.time() - self.time_stop > 4:
            print("3초 지남")
            s_m.data = SERIAL_NOMAL
            self.time_stop = None
            self.check1 = False

    else:
        s_m.data = SERIAL_NOMAL

''' 유턴 미션 함수 '''

def u_turn(self,lidar,local):
    print("-"*20)
    ob = lidar.u_turn_point

    front_cone = None
    if ob is not None and not self.start_uturn:
        if len(ob) != 0:
            for point in ob:
                if self.ob_list.qsize() > 300:
                    self.ob_list.get()
                self.ob_list.put(point)
            
            cones = np.array([c for c in self.ob_list.queue])
            self.clustered_cones = dbscan.np_dbscan(cones,False)

            for clu_cone in self.clustered_cones:
                dist, _ = local.g_kd_tree.query(clu_cone)
                print(dist)
                if dist < 0.85:
                    front_cone = clu_cone
                    self.start_uturn = True
        
    if front_cone is not None:
        print(" 경로 위 콘 있다!")
        self.front_cone = front_cone
            
        map1_g_path = local.g_path
        map1_yaw = local.g_yaw
        map1_k = local.g_k
        map1_kd_tree = local.g_kd_tree
        map2 = local.map_dic[str(self.mission_ind+1)+'.txt']
        map2_g_path = map2.g_path
        map2_yaw = map2.g_yaw
        map2_k = map2.g_k
        map2_kd_tree = map2.g_kd_tree

        map1_near_with_front_cone = map1_g_path[map1_kd_tree.query(self.front_cone)[1]]
        map2_near_with_front_cone = map2_g_path[map2_kd_tree.query(self.front_cone)[1]]

        mid_cone_point = [(map1_near_with_front_cone[0]+map2_near_with_front_cone[0])/2,(map1_near_with_front_cone[1]+map2_near_with_front_cone[1])/2]
        standard_vector = [map2_near_with_front_cone[0]-map1_near_with_front_cone[0],map2_near_with_front_cone[1]-map1_near_with_front_cone[1]]

        map1_near_with_front_car = map1_g_path[map1_kd_tree.query(local.c_UTM)[1]]
        map2_near_with_front_car = map2_g_path[map2_kd_tree.query(local.c_UTM)[1]]

        c1_vector = [map1_near_with_front_car[0]-mid_cone_point[0],map1_near_with_front_car[1]-mid_cone_point[1]]
        c2_vector = [map2_near_with_front_car[0]-mid_cone_point[0],map2_near_with_front_car[1]-mid_cone_point[1]]

        big_cone_list = []

        for cone in self.clustered_cones:
            mid_2_cone_vector = [cone[0]-mid_cone_point[0],cone[1]-mid_cone_point[1]]
            
            c1_cross_product = c1_vector[0]*mid_2_cone_vector[1] - c1_vector[1]*mid_2_cone_vector[0]
            c2_cross_product = c2_vector[0]*mid_2_cone_vector[1] - c2_vector[1]*mid_2_cone_vector[0]
            
            if c1_cross_product >0 and c2_cross_product <0:
                big_cone_list.append(cone)
        
        dist_cone_with_mid = math.inf
        
        for big_cone in big_cone_list:
            d = math.sqrt((big_cone[0]-mid_cone_point[0])**2+(big_cone[1]-mid_cone_point[1])**2)
            if d < dist_cone_with_mid:
                dist_cone_with_mid = d
                self.near_big_cone = big_cone

        if self.near_big_cone is None:
            self.start_uturn = False
            print(" 뚱콘 없다!")
            return
        
        pre_vec = np.array(standard_vector)
        
        off_vec = np.dot(np.array([[0, -1], [1, 0]]), pre_vec)
        norm_off_vec = off_vec / np.linalg.norm(off_vec)
        
        off_point = np.array(self.near_big_cone) + norm_off_vec * 3 # 1.5미터 오프셋 함.
            
        # ======================= 글로벌 패스 슬라이싱 하고 vstack해야 함. =================================================================

        print(self.near_big_cone)
        map1_ind = map1_kd_tree.query(self.near_big_cone)[1]
        map2_ind = map2_kd_tree.query(self.near_big_cone)[1]
        sliced_map1 = map1_g_path[map1_ind-2:map1_ind,:]
        sliced_map2 = map2_g_path[map2_ind:map2_ind+2,:]
        path = np.vstack((sliced_map1, off_point, sliced_map2))
        u_turn_path,u_turn_yaw,u_turn_k= generate_target_course(path)

        before_g_path = map1_g_path[:map1_ind-2,:]
        before_g_yaw = map1_yaw[:map1_ind-2]
        before_g_k = map1_k[:map1_ind-2]

        after_g_path = map2_g_path[map2_ind+2:800,:]
        after_g_yaw = map2_yaw[map2_ind+2:800]
        after_g_k = map2_k[map2_ind+2:800]

        local.g_path = np.vstack((before_g_path,u_turn_path,after_g_path))
        local.g_yaw = np.concatenate((before_g_yaw,u_turn_yaw,after_g_yaw))
        local.g_yaw = np.concatenate((before_g_k,u_turn_k,after_g_k))

        local.g_kd_tree = KDTree(local.g_path)
    else:
        print(" 경로 위 콘 없다!")
def slow_drive(self):
    self.c_serial_mode.data = SERIAL_NOMAL

def bagjituck(self):
    print("-"*20)
    print(' 방지턱')
    c_utm = self.local.c_UTM
    dist_with_tuck = math.sqrt((c_utm[0]-TUCK[0])**2+(c_utm[1]-TUCK[1])**2)
    if dist_with_tuck < 10:
        self.c_serial_mode.data = SERIAL_SLOW
    else:
        self.c_serial_mode.data = SERIAL_NOMAL
