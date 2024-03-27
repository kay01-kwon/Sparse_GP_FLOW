import socket
import numpy as np
import struct
import time

from typing import Callable, Tuple
import numpy as np
import tensorflow as tf
import gpflow
import time




PORT = 1029

data = []
data = np.float64(5.1)
byte_array = bytearray(struct.pack("d", data))

save_dir = "saved_model"
loaded_model = tf.saved_model.load(save_dir)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('localhost', PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data_recv = conn.recv(16)
            if data_recv:
                data_recv1 = np.float64(struct.unpack("d", data_recv[0:8]))
                data_recv2 = np.float64(struct.unpack("d", data_recv[8:]))
                X_vstack = np.vstack((data_recv1, data_recv2)).T
                f_mean, f_var = loaded_model.compiled_predict_f(X_vstack)
                data_send = f_mean.numpy()
                print('data1', data_recv1, '+ data2', data_recv2, '= result',data_send)
                byte_array = bytearray(struct.pack("d", data_send))
                conn.sendall(byte_array)
            # time.sleep(1)
    print("Connection closed")
    conn.close()