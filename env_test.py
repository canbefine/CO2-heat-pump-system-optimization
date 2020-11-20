"""
Created by Keven
"""
import numpy as np
import dympy as dy


np.random.seed(1) #指定一个随机种子集


class TmsEnv(object):
    # viewer = None
    action_bound = [[80., 20., 1000, 200], [120., 100., 3000, 500]] # 压力从80Bar 到 130Bar；压缩机转速从20Hz 到60Hz
    state_dim = 2
    action_dim = 4
    t_target = 20  # 目标舱内温度
    w = 0.5   # Reward中，舒适性：能耗 = 1：0.5
    stoptime = 120    # 仿真停止时间(s)
    airTemp_bound = [273.15 - 20., 273.15 + 20.]

    dymola = dy.Dymola()

    def __init__(self):
        self.dymola.clear() # 清理dymola进程，这步可去除
        # 打开绝对路径中的dymola文件
        self.dymola.openModel('C:\\Users\\c00500953\\PycharmProjects\\example1\\Dymola\\examples\\CO2\\CO2_2.mo')
        # 编译对应的Dymola文件
        self.dymola.compile('CO2_2')
        # 定义并初始化一些变量和函数
        self.t_air_amb = round(35 + 273.15)
        self.t_air_inCar = round(20 + 273.15)
        self.normalize(273.15+20., 273.15-20., 273.15+15.)
        # self.readTxt()

    def step(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        print('action: ',action) # 打印action，供观察


        ### 开始向Dymola赋值，需要打开Dymola
        # self.dymola.set_parameters({'p_dis_target': 1e5 * action[0], 'n_comp': action[1]})
        self.dymola.write_dsu({'time': [0],
                               'p_dis_target': [    1e5 * action[0]   ],
                               'n_comp': [  action[1]     ],
                               'v_flow_fan': [  action[2]/3600.   ],
                               'v_flow_air': [  action[3]/3600.   ],
                               'ratio_rec': [  0.9   ],
                               })
        self.dymola.simulate(StopTime=self.stoptime,Tolerance=1e-3)
        ### Dymola仿真结束


        res = self.dymola.get_result() # 从仿真文件中取值
        length = int(np.floor(len(res['evaporator.summary.T_air_B']) / 2))
        s_dict = {'t_air': res['evaporator.summary.T_air_B'][-1] - 273.15,
                  'P_dis': res['scrollCompressor.summary.p_B'][-1] / 1e5,
                  'P_suc': res['scrollCompressor.summary.p_A'][-1] / 1e5,
                  'SH': res['sensor_superheating.sensorValue'][-1],
                  'P_ratio': (res['scrollCompressor.summary.p_B'][-1] / res['scrollCompressor.summary.p_A'][-1]),
                  't_dis': res['scrollCompressor.summary.T_B'][-1] - 273.15,
                  'E_comp': res['scrollCompressor.summary.P_shaft'][-1] / 1000,
                  'Clocktime': res['clock.y'][-1],
                  't_air_inCar': res['sensor_T_recAir.sensorValue'][-1]  # cel degree
                  }

        s_ = list(s_dict.values()) # 将从仿真文件取的值放入list中

        print('state: ', s_) # 打印，供观察

        # 告警后停止episode，返回较差的reward
        if s_[1] > 140 or s_[2] < 10 or s_[3] < 0 or s_[4] > 12 or s_[5] > 165 or s_[7] < self.stoptime:
            reward = -10.
            done = True
        # 如无告警，返回正常的reward
        else:
            reward = 5 * (
                    -self.normalize(10., 0., abs(s_[8] - self.t_target))
                    - self.w * self.normalize(10., 0., s_[6] + self.power(action[3],-5.554e-07,0.001507,-0.01895,8.795) + self.power(action[2],2.869e-08,-6.039e-05,0.1186,-50.25))

                          )

        # episode停止条件
        if abs(s_[8] - self.t_target) <= 0.3:
            done = True

        print('reward: ', reward)  # 打印reward

        s__list = [self.normalize(self.airTemp_bound[1], self.airTemp_bound[0],self.t_air_amb),
                    self.normalize(self.airTemp_bound[1], self.airTemp_bound[0],round(s_[8]+273.15))
                  ] # next state
        s__ = np.array(s__list) # 转化为算子

        return s__, reward, done


    def reset(self):
        self.t_air_amb = round(273.15 + 30 + 5 * np.random.rand())
        self.t_air_inCar = round(273.15 + 20 + 5 * np.random.rand())
        # 初始化state
        self.dymola.set_parameters({'t_air_amb': self.t_air_amb, 't_air_inCar': self.t_air_inCar})
        s_list = [self.normalize(self.airTemp_bound[1], self.airTemp_bound[0],self.t_air_amb),
                  # self.normalize(self.airTemp_bound[1], self.airTemp_bound[0],self.t_air_rec),
                  self.normalize(self.airTemp_bound[1], self.airTemp_bound[0],self.t_air_inCar),
                  ]
        s = np.array(s_list) # 转化为算子

        return s
    # def render(self):
    #     if self.viewer is None:
    #         self.viewer = Viewer()
    #     self.viewer.render()
    #     pass

    def normalize(self, max, min, x):
        res = (x - min) / (max - min)

        return res

    def readTxt(self):
        with open(r'C:\Users\c00500953\RL\Lib\site-packages\dympy\dymfiles\dslog.txt', 'r') as f:
            lines = f.readlines()
            for lines in lines:
                if "... Warning message from dymosim" in lines:
                    print('仿真失败')
                    return False
                else:
                    return True

    def power(self, x, p1, p2, p3, p4):
        p = p1 * x ** 3 + p2 * x ** 2 + p3 * x + p4 # unit: kW
        return p/1000

# if __name__ == "__main__":
#
#     # dymola = dy.Dymola()
#     # dymola.clear()
#     # dymola.openModel('C:\\Users\\c00500953\\PycharmProjects\\example1\\Dymola\\examples\\CO2\\CO22222.mo')
#     # dymola.compile('CO22222')
#     env = TmsEnv()
#     # env.reset()
#     while True:
#         env.reset()
#         env.step([110,30])
