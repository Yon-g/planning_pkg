#!/usr/bin/env python3

#ros2 모듈
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from std_msgs.msg import Float64MultiArray, Float64, Int16, Int32MultiArray,Bool
from geometry_msgs.msg import PointStamped

#기본 모듈
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import time
import os
import numpy as np
from queue import Queue

#커스텀 모듈
try:
    from .lib import mission_func as mf
    from . import data
except ImportError:
    raise

#미션 리스트
MISSION_LIST = [0,0,7,26,11,0,23,22,0,21,28,12,21,9,9,22,29,22,30,23,27,21,0,21,0,26,14,0] # final final final final final final final final

#미션 딕셔너리
MISSION_DIC = {0:'고속',7:'배달A',9:'배달B',11:'정적소형',12:'정적대형',14:'평행주차',15:'터널',16:'유턴',17:'터널동적',21:'정지선 직진',22:'정지선 좌회전',23:'정지선 우회전',24:'감속구간',25:'방지턱',26:'중속',27:'중가속',28:'대형전가속',29:'배달후좌회전',30:'배달후좌회전후좌회전후고속'}

#맵 가져올 경로
CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIRECTORY = os.path.dirname(CURRENT_FILE_PATH)
GLOBAL_UTM_TXT_ADDRESS = CURRENT_DIRECTORY + "/map_final/"

#plt 출력 여부
ANI_PRINT = False

def draw_graph(local_path,global_path,car_odom,parking_point,self):
    global count
    plt.clf()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    plt.axis("equal")
    
    ''' 터널 플롯띄울때 쓰는부분 '''
    if self.mission_num.data == 15 or self.mission_num.data == 17:
        if self.first_line is not None:
            plt.plot(self.first_line[:,0],self.first_line[:,1],'or')
        if self.second_line is not None:
            plt.plot(self.second_line[:,0],self.second_line[:,1],'og')
        if self.third_line is not None:
            plt.plot(self.third_line[:,0],self.third_line[:,1],'ob')
        if self.tunnel_path is not None:
            plt.plot(self.tunnel_path[1:,0],self.tunnel_path[1:,1],'ob')
        if self.cen_path is not None:
            plt.plot(self.cen_path[1:,0],self.cen_path[1:,1],'ob')
        if self.wall_path is not None:
            if len(self.wall_path) > 1:
                plt.plot(self.wall_path[1:,0],self.wall_path[1:,1],'ob')
        if self.L_wall is not None:
            plt.plot(self.L_wall_2D[:,0],self.L_wall_2D[:,1],'-r')
        if self.R_wall is not None:
            plt.plot(self.R_wall_2D[:,0],self.R_wall_2D[:,1],'-b')
        if self.lidar.tunnel_small_obj is not None:
            if 2 < len(self.lidar.tunnel_small_obj) < 4:
                plt.plot(self.lidar.tunnel_small_obj[0],self.lidar.tunnel_small_obj[1],'or' )
            elif len(self.lidar.tunnel_small_obj) > 4:
                plt.plot(self.lidar.tunnel_small_obj[0],self.lidar.tunnel_small_obj[1],'or' )
                plt.plot(self.lidar.tunnel_small_obj[2],self.lidar.tunnel_small_obj[3],'or' )
    else:
        plt.plot(local_path[0::2],local_path[1::2],'ob')
        plt.plot(global_path[:,0],global_path[:,1],'-r')
        plt.plot(car_odom[0],car_odom[1],'og')

        if self.lidar.points is not None:
            plt.plot(self.lidar.bigcon_point[:,0],self.lidar.bigcon_point[:,1],'or')            
        if parking_point is not None:
            plt.plot(parking_point[2][0],parking_point[2][1],'oy')
        if self.f_target_p[0] != 0:
            plt.plot(self.f_target_p[0],self.f_target_p[1],'or')
        if self.s_target_p[0] != 0:
            plt.plot(self.s_target_p[0],self.s_target_p[1],'oy')
        if self.near_big_cone is not None:
            plt.plot(self.near_big_cone[0],self.near_big_cone[1],'og')
        # if self.taget_flag: # 배달 B에서 사용하는 Plot
        #     plt.plot(self.deli_B_g_path[self.path_flag[1],0],self.deli_B_g_path[self.path_flag[1],1],'oc')
        # plt.axis([car_odom[0]-s_size,car_odom[0]+s_size,car_odom[1]-s_size,car_odom[1]+s_size])
        if self.clustered_cones is not None:
            c = np.array(self.clustered_cones)
            plt.plot(c[:,0],c[:,1],'or')
    plt.grid(True)
    plt.pause(0.01)

class Planning(Node):

    def __init__(self):
        super().__init__('final', namespace='/Planning/core')
        qos_profile_action = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,depth=1,reliability=QoSReliabilityPolicy.BEST_EFFORT,durability=QoSDurabilityPolicy.VOLATILE)
        qos_profile_sensor = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,depth=5,reliability=QoSReliabilityPolicy.BEST_EFFORT,durability=QoSDurabilityPolicy.VOLATILE)

        """ 데이터 객체 생성 """
        self.lidar = data.Lidar()
        self.vision = data.Vision()
        self.local = data.Local(MISSION_LIST,MISSION_DIC,GLOBAL_UTM_TXT_ADDRESS)
        self.control = data.Control()
        #-------------------------------------Pub and Sub-----------------------------------------

        self.lo_path_pub = self.create_publisher(Float64MultiArray,'/Planning/local_path', qos_profile_sensor) #로컬패스
        self.lo_yaw_pub = self.create_publisher(Float64MultiArray,'/Planning/path_yaw', qos_profile_sensor) #로컬yaw 
        self.lo_k_pub = self.create_publisher(Float64MultiArray,'/Planning/curvature', qos_profile_sensor) #로컬k
        self.c_serial_mode_pub = self.create_publisher(Int16,'/Planning/control_switch', qos_profile_action) # 시리얼 번호
        self.mission_flag_pub = self.create_publisher(Int16,'/Planning/mission',qos_profile_action) # 미션번호

        self.v_flag_done_pub = self.create_publisher(Int16,'/Planning/deli_flag',qos_profile_sensor) # flag mission done check sub 배달
        self.car_yaw_pub = self.create_publisher(Float64,'/Planning/heading',qos_profile_sensor) # 터널에서 차량 heading publishe

        """ Local """
        self.c_utm_sub = self.create_subscription(PointStamped,'/Local/utm',self.local.lo_c_UTM_callback,qos_profile_sensor) # utm sub
        self.c_yaw_sub = self.create_subscription(Float64,'/Local/heading',self.local.lo_c_yaw_callback,qos_profile_sensor)  # yaw sub

        """ Lidar """
        self.l_dynamic_sub = self.create_subscription(Bool,'/LiDAR/dynamic_stop',self.lidar.dynamic_callback,qos_profile_action) # 동적 sub
        self.l_small_object_sub = self.create_subscription(Float64MultiArray,'/Planning/small_object_UTM',self.lidar.cone_UTM_callback,qos_profile_sensor) # 정적 소형 sub
        self.l_bigObject_sub = self.create_subscription(Float64MultiArray,'/Planning/big_object_UTM',self.lidar.bigcon_callback,qos_profile_sensor)
        self.prl_point_UTM_sub = self.create_subscription(Float64MultiArray, '/Planning/prl_points', self.lidar.prl_point_UTM_callback,qos_profile_sensor) #평행주차 점 sub
        self.u_turn_point_sub = self.create_subscription(Float64MultiArray,'/Planning/u_turn_point',self.lidar.u_turn_UTM_callback,qos_profile_sensor)

        self.tunnel_small_sub = self.create_subscription(Float64MultiArray,'/LiDAR/object_cen', self.lidar.tunnel_small_callback,qos_profile_sensor)
        self.deli_point_UTM_sub = self.create_subscription(Float64MultiArray, '/Planning/deli_UTM', self.lidar.deli_point_UTM_callback ,qos_profile_sensor) #배달 point sub
        self.prl_stop_sub = self.create_subscription(Bool, '/LiDAR/park_ok2', self.lidar.prl_stop_callback,qos_profile_action) #평행주차 점 sub

        self.tunnel_wall_sub = self.create_subscription(Float64MultiArray, '/LiDAR/wall_dist', self.lidar.ternnel_wall_callback,qos_profile_sensor) # 터널 벽 sub
        self.tunnel_lane_sub = self.create_subscription(Float64MultiArray, '/LiDAR/lane', self.lidar.ternnel_lane_callback,qos_profile_sensor) # 차선 점 sub
        self.tunnel_ceiling_sub = self.create_subscription(Float64MultiArray, '/LiDAR/ceiling_end', self.lidar.ternnel_ceiling_callback,qos_profile_sensor) # 천장 sub
        self.tunnel_center_sub = self.create_subscription(Float64MultiArray, '/LiDAR/center_lane', self.lidar.ternnel_center_callback,qos_profile_sensor) # 차선 중앙점 sub

        """ Vision """
        self.v_traffic_light_sub = self.create_subscription(Int32MultiArray, '/Vision/traffic_sign',self.vision.v_traffic_light_callback,qos_profile_sensor) #traffic light sub
        self.dis_stopline = self.create_subscription(Float64, '/Vision/stopline', self.vision.stopline_callback,qos_profile_action) # 정지선까지의 거리 sub 용승test
        
        """ Control """
        #-----------------------------------------------------------------------------------------

        #tmp_variable 임시 미션 변수
        """공용 임시 변수"""
        self.create_mission_path = False # 미션 안에서 local_path를 새로 만든 경우 기존 경로 생성 방지

        """배달미션 임시 변수"""
        #302491.7248856042,4123762.395569992 kcity flag A 좌표
        self.delivery_A_stop_point = [302491.7248856042,4123762.395569992]
        self.DBUG = True
        self.wait = False
        self.turn = False
        self.start_t = 0
        self.flag_on = 0
        self.fusion_state = Int16()
        self.Deli_flag = None

        self.A_flag_index = []
        self.call_flag_utm_time = None
        self.deli_utm_callback = False

        # -- 배달B 변수 선언 --
        self.taget_flag = False
        self.B_flag = None
        self.deli_B_start = False
        self.chang_g_path_kd = None
        self.path_flag = None

        self.map_change = False
        self.fusion_state.data = 1

        """정지선 임시 변수"""
        self.is_wait = False         
        self.TF_COLOR = 0
        self.stop_l_done = False
        self.start_stopline = False
        self.keep_going = False
        self.go_out = True
        self.c_msg = Int16()
        self.start_t = 0

        """정지선2 임시 변수"""        
        self.stop_ing = False  
        self.stop_ing_2 = False
        self.late_2_stop = False
        self.check = False
        self.stopline_order = -1  # 처음에 몇으로 할지 정해둬야함 중간에 다른미션부터 하면 그 앞에 몇개의 정지선을 건너뛰었는지 생각하고 숫자를 더할 것.(초기값 : -1)!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.tf_time = None
        self.tf_time = None

        """정적소형 임시 변수"""
        self.tmap = False
        self.Tmap = False
        self.f_target_p = [0, 0]
        self.s_target_p = [0, 0]
        self.tae_path = None
        self.standard_path = None

        """정적대형 임시 변수"""
        self.count_big_obj = 0
        self.big_step = 0
        self.real_count = 0
        self.first_near_point_index = None
        
        """터널주행 임시 변수"""
        self.tail_done = False
        self.head_done = False
        self.tunnel_path = None
        self.tunnel_path_ready = False
        self.cen_path = None
        self.term_path = None
        self.term_path_ready = False
        self.tunnel_lane_2D = []
        self.outsider = np.empty((0, 2))
        self.first = np.empty((0, 2))
        self.second = np.empty((0, 2))
        self.third = np.empty((0, 2))
        self.third_order = None
        self.cen_point = None
        self.first_line = None
        self.third_line = None
        self.second_line = None
        self.wall_path = None
        self.L_wall = None
        self.R_wall = None
        self.gg = True
        self.go = None
        self.L_wall_2D = []
        self.R_wall_2D = []
        self.small_ing = False
        self.dynamic_ing = False
        self.time_stop = None
        self.check1 = False
        self.tunnel_small_check = None
        self.change_flag = False
        self.tunnel_dynamic_mission = Int16()
        self.wall_check = False
        self.start_time = None
        self.elapsed_time = 0
        self.in_tunnel = False
        self.ceil_time = None
        self.wall_mode = False
        self.lane_mode = True
        self.gps_mode = False
        self.gps_time = None
        self.change_gps = False

        """유턴 임시변수"""
        self.start_uturn = False
        self.ob_list = Queue()
        self.front_cone = None
        self.near_big_cone = None
        self.clustered_cones = None


        """평행주차 임시변수"""
        self.prl_park_path = None
        self.prl_st_point = None
        self.prl_st_rad = None
        self.prl_st_vec = None
        self.prl_path = None
        self.prl_yaw = None
        self.prl_k = None
        self.prl_tree = None
        self.prl_R1 = 3.0 # 작은원 (안쪽)            튜닝필수!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.prl_R2 = 3.5 # 큰원 (바깥쪽)          튜닝필수!!!!!!!!!!!!!!!!!!!!!!!!!!             
        self.offset_D = 1.5 # 스타트점 세로로 offset하는 값       튜닝필수!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.offset_short = 0 # 가로로 띄움      쓰지마ㅋㅋ
        self.offset_gap = 1.1 # 호 두개 띄움     튜닝필수!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.esc_D = 6.5 # 탈출경로 만드는 거리 (m)  튜닝필수!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.for_stop_D = 15 # 정지 예비 포인트 찍을 거리 (m)
        self.time_check = False
        self.prl_space = None
        self.daeun_path = None
        self.ready_path_1 = False
        self.ready_path_2 = False
        self.ready_path_3 = False
        self.ready_path_4 = False
        self.finish_path_1 = False
        self.finish_path_2 = False
        self.prl_total = None
        self.prl_ready_path = None
        self.prl_ready_yaw = None
        self.prl_ready_k = None
        self.created_check = False
        self.prl_fin_point = None
        self.escape_path = None
        self.escape_path_pre = None
        self.esc_path = None
        self.esc_yaw = None
        self.esc_k = None
        self.esc_tree = None
        self.esc_path_ready = False
        self.finish_esc_path = False
        self.out_nearest_index = None
        
        """메세지 선언"""
        self.l_path = Float64MultiArray() # 경로
        self.l_yaw = Float64MultiArray() # 기울기
        self.l_k = Float64MultiArray() # 곡률
        self.car_yaw = Float64() # 터널에서 차량 heading
        self.c_serial_mode = Int16() # 시리얼 모드 [ 0 : nomal mode, 2 : E-Stop, 3 : Encoder mode ]
        self.mission_num = Int16() # 미션 번호
        self.c_serial_mode.data = 0 # 시리얼 모드 초기화

        while(not self.local.lo_c_UTM_callback_called) : print("utm 안들어옴"); time.sleep(0.05); rclpy.spin_once(self)
        while(not self.local.lo_c_yaw_callback_called) : print("heading 안들어옴"); time.sleep(0.05); rclpy.spin_once(self)

        self.mission_ind = self.local.chose_now_mission_map()
        self.timer_period = 0.01  # hz100 건드리지 마세요
        self.timer = self.create_timer(self.timer_period, self.planning)

    def planning(self):
        t1 = time.time()
        self.create_mission_path = False
        if self.local.is_mission_complete(): # 미션이 끝나가는지? 끝나가면 True, 아직이면 False
            print("이번 미션 끝남")
            if self.mission_ind == len(MISSION_LIST) -1: #미션 끝났으면 planning 종료
                print("종료")
                return # 미션 종료!!
            else:
                self.mission_ind+=1
                self.local.change_next_mission(self.mission_ind)
                self.check = False # 정지선 변수 초기화
                self.stop_ing = False
                self.late_2_stop = False
                self.time_check = False
                self.tf_time = None
            
        self.mission_num.data = MISSION_LIST[self.mission_ind]
        print('#'*20)
        print('현재 진행중인 미션 인덱스 :',self.mission_ind)
        print('현재 진행중인 미션 : ',MISSION_DIC[MISSION_LIST[self.mission_ind]])
        if self.mission_num.data == 0:
            self.c_serial_mode.data = 0

        elif self.mission_num.data == 7:
            mf.delivery_ready( self,self.local.c_UTM,self.lidar.deli_flag_UTM )

        elif self.mission_num.data == 9:
            mf.fusion_deli_B(self, self.local.c_UTM)
            #mf.delivery_throw( self, self.local.c_UTM, self.lidar.flagxy, self.vision.v_flag )

        elif self.mission_num.data == 11:
            mf.small_object_fuc(self,self.local,self.lidar,self.c_serial_mode)

        elif self.mission_num.data == 12:
            mf.bbangbbang_path(self,self.local,self.lidar)
        
        elif self.mission_num.data == 14:
            mf.prl_mission_func(self,self.local,self.lidar,self.c_serial_mode)

        elif self.mission_num.data == 15:
            mf.drive_turnel(self,self.lidar,self.c_serial_mode)

        elif self.mission_num.data == 16:
            mf.u_turn(self,self.lidar,self.local)
            
        elif self.mission_num.data == 17:
            mf.drive_turnel(self,self.lidar,self.c_serial_mode)

        elif self.mission_num.data == 21:
            mf.stopline_straight(self,self.local.c_UTM,self.vision.v_light,self.c_serial_mode)
        
        elif self.mission_num.data == 22:
            mf.stopline_left(self,self.local.c_UTM,self.vision.v_light,self.c_serial_mode)

        elif self.mission_num.data == 23:
            mf.stopline_no_light(self,self.local.c_UTM,self.vision.v_light,self.c_serial_mode)

        elif self.mission_num.data == 24:
            #감속구간
            mf.slow_drive(self)

        elif self.mission_num.data == 25:
            #방지턱
            mf.bagjituck(self)

        elif self.mission_num.data == 26:
            #중속
            self.c_serial_mode.data = 4

        elif self.mission_num.data == 27:
            #중가속(우회전 후 가속)
            self.c_serial_mode.data = 0

        elif self.mission_num.data == 28:
            #대형전가속
            self.c_serial_mode.data = 0

        elif self.mission_num.data == 29:
            #배달후좌회전
            self.c_serial_mode.data = 0
        
        elif self.mission_num.data == 30:
            #배달후좌회전후좌회전후 고속
            self.c_serial_mode.data = 0
    
    
        if not self.create_mission_path: #미션 내에서 경로를 만들지 않았을 경우 mission_txt의 글로벌 따라감
            self.l_path.data,self.l_yaw.data,self.l_k.data = mf.create_local_path(self.local.c_UTM,
                                                                                    self.local.g_path,
                                                                                    self.local.g_yaw,
                                                                                    self.local.g_k,
                                                                                    self.local.g_kd_tree)
        self.publish_to_control()
        if ANI_PRINT:
            draw_graph(self.l_path.data,self.local.g_path,self.local.c_UTM,self.vision.obq_parking_points,self)

    def publish_to_control(self): # 제어에 publish하는 함수
        self.lo_path_pub.publish(self.l_path)
        self.lo_yaw_pub.publish(self.l_yaw)
        self.lo_k_pub.publish(self.l_k)
        self.c_serial_mode_pub.publish(self.c_serial_mode)
        self.mission_flag_pub.publish(self.mission_num)
        
        if self.mission_num.data == 15: # 터널
            self.car_yaw_pub.publish(self.car_yaw)

        if self.mission_num.data == 7 or self.mission_num.data == 9 : #배달 왼료 pub
            self.v_flag_done_pub.publish(self.fusion_state)
        

def main(args=None):
    rclpy.init(args =args)
    node = Planning()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ =='__main__':
    main()
