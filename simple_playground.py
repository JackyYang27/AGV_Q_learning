import random as r
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from simple_geometry import *
import numpy as np
from PyQt5 import QtWidgets, QtCore
matplotlib.use('Qt5Agg')


class Car():
    def __init__(self) -> None:
        self.diameter = 6
        self.angle_min = -90
        self.angle_max = 270
        self.wheel_min = -40
        self.wheel_max = 40
        self.xini_max = 4.5
        self.xini_min = -4.5

        self.reset()

    @property
    def radius(self):
        return self.diameter/2

    # reset the car to the beginning line
    def reset(self):
        self.angle = 90
        self.wheel_angle = 0

        xini_range = (self.xini_max - self.xini_min - self.diameter)
        left_xpos = self.xini_min + self.diameter//2
        self.xpos = r.random()*xini_range + left_xpos  # random x pos [-3, 3]
        self.ypos = 0

    def setWheelAngle(self, angle):
        self.wheel_angle = angle if self.wheel_min <= angle <= self.wheel_max else (
            self.wheel_min if angle < self.wheel_min else self.wheel_max)

    def setPosition(self, newPosition: Point2D):
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    # this is the function returning the coordinate on the right, left, front or center points
    def getPosition(self, point='center') -> Point2D:
        if point == 'right':
            right_angle = self.angle - 45
            right_point = Point2D(self.diameter/2, 0).rotate(right_angle)
            return Point2D(self.xpos, self.ypos) + right_point

        elif point == 'left':
            left_angle = self.angle + 45
            left_point = Point2D(self.diameter/2, 0).rotate(left_angle)
            return Point2D(self.xpos, self.ypos) + left_point

        elif point == 'front':
            fx = m.cos(self.angle/180*m.pi)*self.diameter/2+self.xpos
            fy = m.sin(self.angle/180*m.pi)*self.diameter/2+self.ypos
            return Point2D(fx, fy)
        else:
            return Point2D(self.xpos, self.ypos)

    def setAngle(self, new_angle):
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min
        self.angle = new_angle

    # set the car state from t to t+1
    def tick(self):
        car_angle = self.angle/180*m.pi
        wheel_angle = self.wheel_angle/180*m.pi
        new_x = self.xpos + m.cos(car_angle+wheel_angle) + \
            m.sin(wheel_angle)*m.sin(car_angle)

        new_y = self.ypos + m.sin(car_angle+wheel_angle) - \
            m.sin(wheel_angle)*m.cos(car_angle)
        new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) / self.diameter)) / m.pi * 180

        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min

        self.xpos = new_x
        self.ypos = new_y
        self.setAngle(new_angle)

class Playground():
    def __init__(self):
        # read path lines
        self.path_line_filename = "軌道座標點.txt"
        self._setDefaultLine()
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),  # start line
            Line2D(0, 0, 0, -3),  # middle line
        ]
        self.q_table_path = "q_table.npy"  # 修改为 NumPy 文件格式
        self.load_q_table()
        self.complete = False
        self.previous_state = [0, 0, 0]
        self.current_state = [0, 0, 0]
        self.previous_angle = 0
        self.current_angle = 0
        # 各有17個，因為我以5度為一個單位
        self.q_table = {
            "close_left": [0]*17,
            "close_center": [0]*17,
            "close_right": [0]*17,
            "far_left": [0]*17,
            "far_center": [0]*17,
            "far_right": [0]*17,
        }
        self.car = Car()
        self.reset()
        self.cumulated_reward = 0
        self.error_count = 0
    def load_q_table(self):
        """尝试加载已保存的 Q-table"""
        if os.path.exists(self.q_table_path):
            self.q_table = np.load(self.q_table_path, allow_pickle=True).item()
            print("Loaded Q-table from file.")
        else:
            # 初始化 Q-table
            self.q_table = {
                "close_left": np.zeros(17),
                "close_center": np.zeros(17),
                "close_right": np.zeros(17),
                "far_left": np.zeros(17),
                "far_center": np.zeros(17),
                "far_right": np.zeros(17),
            }
    def _setDefaultLine(self):
        # print('use default lines')
        # default lines
        self.destination_line = Line2D(18, 40, 30, 37)

        self.lines = [
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]

        self.car_init_pos = None
        self.car_init_angle = None

    def save_q_table(self):
        """保存 Q-table 到文件"""
        np.save(self.q_table_path, self.q_table)
        print("Q-table saved to file.")
    def _readPathLines(self):
        try:
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # get init pos and angle
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]

                # get destination line
                dp1 = Point2D(*[float(v) for v in lines[1].split(',')])
                dp2 = Point2D(*[float(v) for v in lines[2].split(',')])
                self.destination_line = Line2D(dp1, dp2)

                # get wall lines
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception:
            self._setDefaultLine()

    @property
    def n_actions(self):  # action = [0~num_angles-1]  17
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def observation_shape(self):
        return (len(self.state),)

    @ property
    def state(self):
        front_dist = - 1 if len(self.front_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.front_intersects[0])
        right_dist = - 1 if len(self.right_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.right_intersects[0])
        left_dist = - 1 if len(self.left_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.left_intersects[0])

        return [front_dist, right_dist, left_dist]

    def _checkDoneIntersects(self):
        if self.done:
            return self.done

        cpos = self.car.getPosition('center')     # center point of the car
        cfront_pos = self.car.getPosition('front')
        cright_pos = self.car.getPosition('right')
        cleft_pos = self.car.getPosition('left')
        radius = self.car.radius

        isAtDestination = cpos.isInRect(
            self.destination_line.p1, self.destination_line.p2
        )
        # if we finish the tour
        done = False if not isAtDestination else True
        self.complete = False if not isAtDestination else True

        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True
        for wall in self.lines:  # check every line in play ground
            dToLine = cpos.distToLine2D(wall)
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos-p1).length, (cpos-p2).length
            wall_len = wall.length

            # touch conditions
            p1_touch = (dp1 < radius)
            p2_touch = (dp2 < radius)
            body_touch = (
                dToLine < radius and (dp1 < wall_len and dp2 < wall_len)
            )
            front_touch, front_t, front_u = Line2D(
                cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(
                cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(
                cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch or front_touch:
                if not done:
                    done = True

            # find all intersections
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1)*front_u+p1
                if front_t:
                    if front_t > 1:  # select only point in front of the car
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # if overlapped, don't select any point
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1)*right_u+p1
                if right_t:
                    if right_t > 1:  # select only point in front of the car
                        right_intersections.append(right_inter_point)
                    elif right_touch:  # if overlapped, don't select any point
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1)*left_u+p1
                if left_t:
                    if left_t > 1:  # select only point in front of the car
                        left_intersections.append(left_inter_point)
                    elif left_touch:  # if overlapped, don't select any point
                        left_intersections = []
                        find_left_inter = False

        self._setIntersections(front_intersections,
                               left_intersections,
                               right_intersections)

        # results
        self.done = done
        return done

    def _setIntersections(self, front_inters, left_inters, right_inters):
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('left')))

    def reset(self):
        self.done = False
        self.complete = False
        self.car.reset()

        if self.car_init_angle and self.car_init_pos:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)
        
        self._checkDoneIntersects()
        self.cumulative_reward = 0
        return self.state

    def setCarPosAndAngle(self, position: Point2D = None, angle=None):
        if position:
            self.car.setPosition(position)
        if angle:
            self.car.setAngle(angle)

        self._checkDoneIntersects()

    def calWheelAngleFromAction(self, action):
        angle = self.car.wheel_min + \
            action*(self.car.wheel_max-self.car.wheel_min) / \
            (self.n_actions-1)
        return angle

    def step(self, action=None):
        if action:
            # 使用絕對方向取代相對方向的概念
            # E.g.原先方向盤向右轉45度，若下一次向左轉45度
            # 相對方向:0度向前；絕對方向:向左45度
            self.car.setWheelAngle(action)

        if not self.done:
            self.car.tick()
            self._checkDoneIntersects()
            return self.state
        else:
            return self.state

    # relationship function
    def q_table_state(self, car_state):
        f_dis, r_dis, l_dist = car_state
        rl_dif = r_dis-l_dist

        q_state = ""
        if f_dis >= 12:
            q_state += "far_"
        else:
            q_state += "close_"

        if rl_dif > 2.5:
            q_state += "right"
        elif -2.5 <= rl_dif <= 2.5:
            q_state += "center"
        elif rl_dif < -2.5:
            q_state += "left"

        return q_state


    # reward function
    def reward(self, q_state, angle):
        if self.done:
            if self.complete:
                return 10
            else:
                return -10
        if q_state == "close_right":
            if 40 >= angle > 30:
                return 7
            elif 30 >= angle > 25:
                return 4
            elif 25>= angle >20:
                return 3
            elif 20 >= angle > 0:
                return 2
            else:
                return -5
        elif q_state == "close_center":
            if 10 > angle > -10:
                return -4
            else:
                return 1
        elif q_state == "close_left":
            if 0 > angle >= -20:
                return 2
            elif -20 > angle >= -30:
                return 3
            elif -30 > angle >= -40:
                return 4
            else:
                return -5
        if q_state == "far_right":
            if 40 >= angle >= 20:
                return -1
            elif 20 > angle > 0:
                return 1
            else:
                return -3
        elif q_state == "far_center":
            if 10 > angle > -10:
                return 4
            elif 20 >= angle >= 10:
                return -3
            elif -10 >= angle >= -20:
                return -3
            else:
                return -3
        elif q_state == "far_left":
            if 0 > angle > -20:
                return 2
            elif -20 >= angle >= -40:
                return -1
            else:
                return -4


    # update the q_table
    def update_q_table(self, current_state, current_angle, previous_state, previous_angle, a=1, r=1):
        reward = self.reward(current_state, current_angle)
        self.cumulated_reward += reward
        self.q_table[previous_state][self.angle_to_index(previous_angle)] \
            += + a * (self.reward(current_state, current_angle) + r * max(self.q_table[current_state]) -
                      self.q_table[previous_state][self.angle_to_index(previous_angle)])
        print(f"Cumulative Reward and Reward: {self.cumulated_reward},{reward}")
        print("Updated Q-Table:", self.q_table)
    
    # turning index to wheel angle
    def index_to_angle(self, index):
        return -40+index*5
    # turning wheel angle to index
    def angle_to_index(self, angle):
        return (angle+40)//5

    # e-greedy algorithm
    def e_greedy(self, e, q_state):
        if r.random() <= e:
            return self.index_to_angle(self.choose_action(q_state))
        else:
            return self.index_to_angle(r.choice([i for i in range(len(self.q_table[q_state]))]))

    # choose the max score
    def choose_action(self, q_state):
        max_value = max(self.q_table[q_state])
        max_indices = [i for i, v in enumerate(self.q_table[q_state]) if v == max_value]
        return r.choice(max_indices)

    # training model
    def q_learning_training(self, training_time, e):
        for i in range(training_time):
            state = self.reset()
            q_state = self.q_table_state(state)
            e_train = m.exp(-abs(training_time - i + 1)) * e
            while not self.done:
                action = self.e_greedy(e_train, q_state)
                self.previous_state = q_state
                self.previous_angle = self.car.wheel_angle
                self.current_state = self.q_table_state(self.step(action))
                self.current_angle = self.car.wheel_angle
                self.update_q_table(self.current_state, self.current_angle,
                                    self.previous_state, self.previous_angle)
                q_state = self.q_table_state(self.state)
            
            # 检查是否结束但未完成
            if self.done and not self.complete:
                self.error_count += 1  # 增加错误计数
            self.save_q_table()    
        print(f"Training completed with {self.error_count} errors.")

    # actual running model(e是e-greedy用的機率)
    def run_simulation(self, e, state):
        q_state = self.q_table_state(state)
        action = self.e_greedy(e, q_state)
        self.previous_state = q_state
        self.previous_angle = action
        self.current_state = self.q_table_state(self.step(action))
        self.current_angle = self.car.wheel_angle
        self.update_q_table(self.current_state, self.current_angle,
                            self.previous_state, self.previous_angle)

class Animation(QtWidgets.QMainWindow):
    '''
    play: playground的建立
    state: 當前狀態
    now_running: 當下程式是否在執行
    QtCore.QTimer: 控制動畫的執行頻率和狀態
    '''
    def __init__(self, play: Playground):
        super().__init__()
        self.play = play
        self.state = self.play.reset()
        self.now_running = False
        self.timer = QtCore.QTimer(self)

        # 建立主視窗
        self.setWindowTitle("操作介面")
        self.main_widget = QtWidgets.QWidget(self)
        # 將主視窗的中間介面設為main_widget
        self.setCentralWidget(self.main_widget)

        # 開始動畫按鈕
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.start_animation)
        # 停止動畫按鈕
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_animation)
        # 建立繪畫動畫的區域
        # 動畫的底板
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)

        # 建立畫面配置
        # 主要畫面
        layout = QtWidgets.QVBoxLayout(self.main_widget)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.canvas)

        # 事前的背景及變數設定
        self.setup_animation()

    def setup_animation(self):
        '''
        ax: 動畫的畫布
        background: 動畫背景
        start_line: 起點
        end_line: 終點
        car_trail: 車子的移動軌跡
        direction_line: 顯示車子當前方向的直線
        text: 當下感測器所偵測到的距離
        '''

        self.ax = self.figure.add_subplot(111)
        self.background = self.play.lines
        self.start_line = self.play.decorate_lines[0]
        self.end_line = self.play.destination_line
        self.car_trail = []
        self.car_radius = self.play.car.radius
        self.direction_line, = self.ax.plot([], [], 'r-')  # 建立一調指引方向的線
        self.text = self.ax.text(15, 0, '', fontsize=10)

        self.draw_background()

    def draw_background(self):
        for line in self.background:
            self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], "k-")

        # 因為終點是一個矩形，所以需要兩條線
        self.ax.plot([self.end_line.p1.x, self.end_line.p2.x],
                     [self.end_line.p1.y, self.end_line.p1.y], "r-")
        self.ax.plot([self.end_line.p1.x, self.end_line.p2.x],
                     [self.end_line.p2.y, self.end_line.p2.y], "r-")

        # 起點
        self.ax.plot([self.start_line.p1.x, self.start_line.p2.x],
                     [self.start_line.p1.y, self.start_line.p2.y], "b-")

        self.ax.axis('equal')

    # 初始化各項變數後開始動畫
    def start_animation(self):
        if self.now_running:
            self.timer.stop()

        self.clean()
        self.play.reset()
        self.now_running = True

        # 更新動畫的函數
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # 每 50 毫秒更新一次
    
    # 停止動畫
    def stop_animation(self):
        self.timer.stop()
        self.now_running = False
        self.clean()

    # 畫面
    def update_animation(self):
        car_pos = self.play.car.getPosition("center")
        self.draw_car(car_pos)
        self.text.set_text(
            f'Front sensor: {self.play.state[0]:.{3}f}\n'
            f'Right sensor: {self.play.state[1]:.{3}f}\n'
            f'Left sensor: {self.play.state[2]:.{3}f}'
        )

        # 抵達終點發布成功訊息
        if self.play.done:
            if self.play.complete:
                self.show_message("Succeeded!")
            # else:
            #     self.show_message("Failed!")
            self.timer.stop()
            self.now_running = False

        self.play.run_simulation(0.7, self.play.state)
        # 畫出所有移動畫面
        self.canvas.draw()

    # 畫出車子
    def draw_car(self, car_pos):
        self.car = plt.Circle((car_pos.x, car_pos.y), self.car_radius, color="green", fill=False)
        self.ax.add_patch(self.car)
        self.car_trail.append(self.car)
        front_sensor = self.play.car.getPosition("front")
        self.direction_line.set_data([car_pos.x, front_sensor.x], [car_pos.y, front_sensor.y])

    # 再次開始的時候，清理之前的車子移動軌跡
    def clean(self):
        for trace in self.ax.patches:
            trace.remove()
        self.car_trail = []

    # 成功訊息
    def show_message(self, message):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setText(message)
        msg_box.exec_()

    # 顯示動畫
    def run(self):
        self.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    playground = Playground()
    GUI = Animation(playground)
    GUI.run()
    QtCore.QTimer.singleShot(100, lambda: playground.q_learning_training(5000, 0.7))   # 訓練 5000 次
    # 啟動 PyQt5 應用程式的事件循環。事件循環是一個無限循環,它會接收並處理來自作業系統的事件，
    # 例如鍵盤輸入、滑鼠移動等。只要應用程式沒有被關閉,事件循環就會一直運行。exec_() 方法會阻
    # 塞主線程,直到應用程式退出為止。
    app.exec_()

