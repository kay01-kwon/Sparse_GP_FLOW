#! /usr/bin/env python
import rospy
import numpy as np
import socket
import struct
from msg_pkg.msg import actual
from std_msgs.msg import Float64

class ROS_SGP_Client:
    def __init__(self, PORT):
        # Set PORT
        self.PORT = PORT
        # Socket creation
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connection
        self.sock.connect(('localhost', PORT))

        self.torque_meas = np.array([0.0,0.0,0.0])
        self.omega_meas = np.array([0.0,0.0,0.0])

        self.tau_fric = np.float64(0)

        self.subscriber = rospy.Subscriber("/actual", actual,self.callback)
        self.publisher = rospy.Publisher("/tau_fric",Float64,queue_size=1)

    def combine_byte_array(self):
        byte_array1 = self.data2byte_array(self.data1)
        byte_array2 = self.data2byte_array(self.data2)
        return byte_array1 + byte_array2

    def send_all_data(self):
        self.sock.send(self.combine_byte_array())

    def data2byte_array(self, data):
        byte_array = bytearray(struct.pack('d',data))
        return byte_array

    def byte2output_data(self):
        data_recv = self.sock.recv(8)
        data_recv = np.float64(struct.unpack('d',data_recv))
        print(data_recv)
        self.tau_fric = data_recv

    def callback(self, msg):
        self.data1 = msg.act_LIFT_torque[0]
        self.data2 = msg.act_LIFT_vel[0]
        self.send_all_data()
        self.byte2output_data()
        self.publish_tau_fric()
    
    def connection_close(self):
        self.sock.close()

    def publish_tau_fric(self):
        self.publisher.publish(self.tau_fric)

if __name__=='__main__':
    rospy.init_node("SGP_client",anonymous=True)
    ros_SGP_client = ROS_SGP_Client(4000)
    r = rospy.Rate(50)
    while not rospy.is_shutdown():
        r.sleep()

    ros_SGP_client.connection_close()
