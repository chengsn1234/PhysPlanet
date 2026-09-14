#!/usr/bin/env python3
"""
ROS2 键盘控制节点 - 标准版
与 teleop_twist_keyboard 包完全兼容
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import termios
import tty

# 保存原始终端设置
settings = termios.tcgetattr(sys.stdin)

# 标准 teleop_twist_keyboard 消息
msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >

t : up (+z)
b : down (-z)

anything else : stop

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

CTRL-C to quit
"""

# 标准运动按键映射 (线速度x, 线速度y, 线速度z, 角速度z)
moveBindings = {
    'i': (1, 0, 0, 0),
    'o': (1, 0, 0, -1),
    'j': (0, 0, 0, 1),
    'l': (0, 0, 0, -1),
    'u': (1, 0, 0, 1),
    ',': (-1, 0, 0, 0),
    '.': (-1, 0, 0, 1),
    'm': (-1, 0, 0, -1),
    'O': (1, -1, 0, 0),
    'I': (1, 0, 0, 0),
    'J': (0, 1, 0, 0),
    'L': (0, -1, 0, 0),
    'U': (1, 1, 0, 0),
    '<': (-1, 0, 0, 0),
    '>': (-1, -1, 0, 0),
    'M': (-1, 1, 0, 0),
    't': (0, 0, 1, 0),
    'b': (0, 0, -1, 0),
}

# 标准速度调节按键映射
speedBindings = {
    'q': (1.1, 1.1),
    'z': (0.9, 0.9),
    'w': (1.1, 1),
    'x': (0.9, 1),
    'e': (1, 1.1),
    'c': (1, 0.9),
}


class TeleopTwistKeyboard(Node):
    """标准键盘控制ROS2节点"""
    
    def __init__(self):
        super().__init__('teleop_twist_keyboard')
        
        # 创建发布器
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # 初始化速度参数
        self.speed = 1.0
        self.turn = 0.5
        
        self.get_logger().info('键盘控制节点已启动')
    
    def getKey(self):
        """获取按键输入"""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key
    
    def vels(self, speed, turn):
        """显示当前速度"""
        return "currently:\tspeed %s\tturn %s " % (speed, turn)
    
    def run(self):
        """主运行循环"""
        x = 0
        y = 0
        z = 0
        th = 0
        status = 0
        
        try:
            print(msg)
            print(self.vels(self.speed, self.turn))
            
            while True:
                key = self.getKey()
                
                if key in moveBindings.keys():
                    x = moveBindings[key][0]
                    y = moveBindings[key][1]
                    z = moveBindings[key][2]
                    th = moveBindings[key][3]
                
                elif key in speedBindings.keys():
                    self.speed = self.speed * speedBindings[key][0]
                    self.turn = self.turn * speedBindings[key][1]
                    
                    print(self.vels(self.speed, self.turn))
                    if (status == 14):
                        print(msg)
                    status = (status + 1) % 15
                
                else:
                    # 如果是空键且机器人已停止，跳过
                    if key == '' and x == 0 and y == 0 and z == 0 and th == 0:
                        continue
                    
                    x = 0
                    y = 0
                    z = 0
                    th = 0
                    
                    if (key == '\x03'):
                        break
                
                # 发布Twist消息
                twist = Twist()
                twist.linear.x = x * self.speed
                twist.linear.y = y * self.speed
                twist.linear.z = z * self.speed
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = th * self.turn
                
                self.publisher.publish(twist)
        
        except Exception as e:
            print(e)
        
        finally:
            # 发送停止指令
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        teleop_node = TeleopTwistKeyboard()
        teleop_node.run()
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        try:
            teleop_node.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
