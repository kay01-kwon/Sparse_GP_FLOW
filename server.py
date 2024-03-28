import os
import socket
import numpy as np
import struct
import time

from typing import Callable, Tuple
import numpy as np
import tensorflow as tf
import gpflow
import time


class SGPserver:
    def __init__(self, saved_dir, PORT = 4000):

        # Get the saved directory
        self.saved_dir = saved_dir
        print(self.saved_dir)

        # Load the saved model
        self.loaded_model = self.load_saved_model()
        # Sets PORT
        self.PORT = PORT

        self.conn, self.addr = self.socket_handler()



    def load_saved_model(self):
        saved_model = tf.saved_model.load(self.saved_dir)
        return saved_model

    def socket_handler(self):
        # Socket creation
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket created")

        # Socket bind
        sock.bind(('localhost', self.PORT))
        print("Socket bind complete")

        # Socket listen
        sock.listen()
        print("Socket listen")

        # Socket accept
        conn, addr = sock.accept()
        print("Connected by", addr)

        return conn, addr

    def SGPR_run(self):
        while True:
            input_data = self.conn.recv(16)
            if input_data:
                tau_meas, omega_meas = self.get_input(input_data)
                tau_fric_mean, tau_fric_var = self.get_tau_fric(tau_meas, omega_meas)
                self.send_tau_fric(tau_fric_mean, tau_fric_var)
        self.conn.close()


    def get_input(self, input_data):
        """
        Gets input data from the client and convert it to numpy float64 format.
        :param input_data: byte array of tau_meas and omega_meas
        :return: tau_meas, omega_meas
        """
        tau_meas = np.float64(struct.unpack('d', input_data[0:8]))
        oemga_meas = np.float64(struct.unpack('d', input_data[8:]))
        print("tau_meas: ", tau_meas, "oemga_meas: ",oemga_meas)
        return tau_meas, oemga_meas

    def get_tau_fric(self, tau_meas, oemga_meas):
        """
        From the saved SGPR model, get the mean and variance of the tau_fric, respectively.
        :param tau_meas:
        :param oemga_meas:
        :return: tau_fric_mean, tau_fric_var
        """
        X_vstack = np.vstack((tau_meas, oemga_meas)).T
        tau_fric_mean, tau_fric_var = self.loaded_model.compiled_predict_f(X_vstack)
        return tau_fric_mean, tau_fric_var

    def send_tau_fric(self, tau_fric_mean, tau_fric_var):
        """
        Converts into byte array and sends it to the client
        :param tau_fric_mean:
        :param tau_fric_var:
        :return:
        """
        tau_fric_mean_ = tau_fric_mean.numpy()
        tau_fric_var_ = tau_fric_var.numpy()
        print("tau_fric_mean: ",tau_fric_mean_, "tau_fric_var:", tau_fric_var_)
        bytearray_ = bytearray(struct.pack('d',tau_fric_mean_))
        self.conn.sendall(bytearray_)


if __name__ == "__main__":
    current_dir = os.getcwd()
    saved_dir = current_dir + "/saved_model"
    SGP_server = SGPserver(saved_dir)
    SGP_server.SGPR_run()


# PORT = 1029
#
# data = []
# data = np.float64(5.1)
# byte_array = bytearray(struct.pack("d", data))
#
# save_dir = "saved_model"
# loaded_model = tf.saved_model.load(save_dir)
#
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.bind(('localhost', PORT))
#     s.listen()
#     conn, addr = s.accept()
#     with conn:
#         print('Connected by', addr)
#         while True:
#             data_recv = conn.recv(16)
#             if data_recv:
#                 data_recv1 = np.float64(struct.unpack("d", data_recv[0:8]))
#                 data_recv2 = np.float64(struct.unpack("d", data_recv[8:]))
#                 X_vstack = np.vstack((data_recv1, data_recv2)).T
#                 f_mean, f_var = loaded_model.compiled_predict_f(X_vstack)
#                 data_send = f_mean.numpy()
#                 print('data1', data_recv1, '+ data2', data_recv2, '= result',data_send)
#                 byte_array = bytearray(struct.pack("d", data_send))
#                 conn.sendall(byte_array)
#             # time.sleep(1)
#     print("Connection closed")
#     conn.close()