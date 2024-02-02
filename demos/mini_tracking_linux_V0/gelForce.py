import find_marker

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import copy

import numpy as np
import cv2
import time
import marker_detection
import sys
import setting
import os
from gelsight import gsdevice
from gelsight import gs3drecon

import pyqtgraph as pg


A = np.loadtxt(r'BEM/A.dat')  # coefficient matrix
B1 = np.loadtxt(r'BEM/B1.dat', delimiter=',')
U_A = np.loadtxt(r'BEM/U_A.dat', dtype=int)

T_matrix = np.zeros((7, 9))

plot_flag = False


def calcForce(displace_array):
    global A, B1, U_A
    dimension = 3  # 3D problem
    # Solaris
    E = 0.25  # Elastic modulus, unit: MPa
    miu = 0.36  # Poisson's ratio
    # Ecoflex 00-45 Near Clear
    # E = 0.09844  # Elastic modulus, unit: MPa
    # miu = 0.398  # Poisson's ratio
    G = E / (2 * (1 + miu))  # Shear modulus, unit: MPa

    # Solve system equations of BEM
    # {u} displacement array (mm), updated by GelSight in realtime
    u = displace_array  # 改这行！
    # A = np.loadtxt(r'BEM/A.dat')  # coefficient matrix
    # B1 = np.loadtxt(r'BEM/B1.dat', delimiter=',')
    y = np.matmul(B1, u)  # {y}=[B1]*{u}
    x = np.linalg.solve(A, y)  # solve [A]*{x}={y}

    # Prepare a tool array to retrieve tractions of interest based on global nodal numbers
    # U_A contains global nodal number for each marker
    # U_A = np.loadtxt(r'BEM/U_A.dat', dtype=int)
    selectArray = np.zeros(len(U_A) * dimension, dtype=int)  # the tool array
    for i in range(len(U_A)):
        for j in range(dimension):
            id = dimension * i + j
            selectArray[id] = dimension * (U_A[i] - 1) + j

    # Retrieve tractions of interest from {x}
    # T contains Tx, Ty, Tz for each node of interest
    T = x[selectArray] * 2 * G  # unit: MPa

    return T


def restore_Tmatrix(result, rows, cols):
    if not result.any():
        return []

    # matrix = [[0] * cols for _ in range(rows)]
    matrix = np.zeros((rows, cols, 3))
    index = 0

    for start_col in range(cols - 1, -1, -1):
        row, col = 0, start_col
        while row < rows and col < cols:
            matrix[row][col] = result[index:index+3]
            index += 3
            row += 1
            col += 1

    for start_row in range(1, rows):
        row, col = start_row, 0
        while row < rows and col < cols:
            matrix[row][col] = result[index:index+3]
            index += 3
            row += 1
            col += 1

    return matrix

def diagonal_traversal(matrix):
    if not matrix.any():
        return []

    rows, cols = len(matrix), len(matrix[0])
    result = []

    for start_col in range(cols - 1, -1, -1):
        row, col = 0, start_col
        while row < rows and col < cols:
            result.extend(matrix[row][col].tolist())
            row += 1
            col += 1

    for start_row in range(1, rows):
        row, col = start_row, 0
        while row < rows and col < cols:
            result.extend(matrix[row][col].tolist())
            row += 1
            col += 1

    return result


def draw_flow(frame, flow, matrix):
    Ox, Oy, Cx, Cy, Occupied = flow

    dx = np.mean(np.abs(np.asarray(Ox) - np.asarray(Cx)))
    dy = np.mean(np.abs(np.asarray(Oy) - np.asarray(Cy)))
    dnet = np.sqrt(dx**2 + dy**2)
    #print (dnet * 0.075, '\n')


    K = 4
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            pt2 = (int(K*matrix[i][j][0] + Ox[i][j]), int(K*matrix[i][j][1] + Oy[i][j]))
            color = (0, 255, 0)
            if Occupied[i][j] <= -1:
                color = (127, 127, 255)
            cv2.arrowedLine(frame, pt1, pt2, color, 1,  tipLength=0.25)


class HeatMap(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Show")
        self.setGeometry(100, 100, 1200, 800)

        # 创建Matplotlib图形对象
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.res_data = np.zeros((7, 9))

        self.heatmap = self.ax.imshow(self.res_data, cmap='viridis', interpolation='nearest', vmin=-2, vmax=2)
        # self.heatmap.set_clim(vmin=0, vmax=10)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(which='minor', color='black', linewidth=2)

        # 添加颜色条
        self.fig.colorbar(self.heatmap)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.show()
        # self.update()

        self.timer1 = pg.QtCore.QTimer()  # timer的定义必须在__init__中
        self.timer1.timeout.connect(self.plot_data)  # call plotData()
        self.timer1.start(100)

    def show_value(self, force_data):
        for i in range(7):
            for j in range(9):
                value = force_data[i][j]
                self.ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white', fontsize=10)

    def plot_data(self):
        global T_matrix, plot_flag
        if plot_flag:
            self.res_data = T_matrix[:, :, 2]

            self.ax.clear()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw_idle()
            self.ax.imshow(self.res_data, cmap='viridis', interpolation='nearest', vmin=-2, vmax=2)
            self.show_value(self.res_data)


class Reconstruction3D:
    def __init__(self):
        path = '.'
        finger = gsdevice.Finger.MINI
        mmpp = 0.0625

        cam_id = gsdevice.get_camera_id("GelSight Mini")
        dev = gsdevice.Camera(finger, cam_id)
        dev.imgw = 240
        dev.imgh = 320
        net_file_path = 'nnmini.pt'

        ''' Load neural network '''
        model_file_path = path
        net_path = os.path.join(model_file_path, net_file_path)
        print('net path = ', net_path)

        gpuorcpu = "cpu"
        
        self.nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.MINI, dev)
        self.nn.load_nn(net_path, gpuorcpu)
        
        self.vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp)

    def calc_depth(self, frame):
        dm = self.nn.get_depthmap(frame, True)
        self.vis3d.update(dm)
        return dm


def resize_crop_mini(img, imgw, imgh):
    # resize, crop and resize back
    img = cv2.resize(img, (895, 672))  # size suggested by janos to maintain aspect ratio
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
    img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    img = img[:, :-1]  # remove last column to get a popular image resolution
    img = cv2.resize(img, (imgw, imgh))  # final resize for 3d
    return img


def main(argv):
    imgw = 320
    imgh = 240

    calibrate = False

    force_output = False

    outputFileDir = 'res_data/'

    SAVE_DATA_FLAG = True

    # timestr = time.strftime("%Y%m%d_%H%M%S")
    datadir = outputFileDir + 'data'
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    fileNameStr1 = "dispalce_Matrix"
    fileNameStr2 = "T_Matrix"

    datafilename1 = datadir + os.sep + fileNameStr1 + '.txt'
    dispalceMatrixFile = open(datafilename1, "a")

    datafilename2 = datadir + os.sep + fileNameStr2 + '.txt'
    TMatrixFile = open(datafilename2, "a")

    current_timestamp = time.time()
    # 将时间戳转换为本地时间的struct_time对象
    local_time = time.localtime(current_timestamp)
    # 使用strftime()方法将struct_time对象格式化为指定的时间字符串
    current_time = time.strftime("%Y_%m_%d_%H%M%S", local_time)

    Flist_file = "F_result_" + current_time + ".txt"

    if len(sys.argv) > 1:
        if sys.argv[1] == 'calibrate':
            calibrate = True

        if sys.argv[1] == 'force':
            force_output = True


    cap = cv2.VideoCapture(2)
    WHILE_COND = cap.isOpened()

    # set the format into MJPG in the FourCC format
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # Resize scale for faster image processing
    setting.init()
    RESCALE = setting.RESCALE

    counter = 0
    while True:
        if counter == 50:
            ret, frame = cap.read()
            ##########################
            frame = resize_crop_mini(frame, imgw, imgh)
            ### find marker masks
            mask = marker_detection.find_marker(frame)
            ### find marker centers
            mc = marker_detection.marker_center(mask, frame)
            break
        else:
            counter += 1

    mc_sorted1 = mc[mc[:, 0].argsort()]
    mc1 = mc_sorted1[:setting.N_]
    mc1 = mc1[mc1[:, 1].argsort()]

    mc_sorted2 = mc[mc[:, 1].argsort()]
    mc2 = mc_sorted2[:setting.M_]
    mc2 = mc2[mc2[:, 0].argsort()]

    """
    N_, M_: the row and column of the marker array
    x0_, y0_: the coordinate of upper-left marker
    dx_, dy_: the horizontal and vertical interval between adjacent markers
    """

    N_ = setting.N_
    M_ = setting.M_
    fps_ = setting.fps_
    x0_ = np.round(mc1[0][0])
    y0_ = np.round(mc1[0][1])
    dx_ = mc2[1, 0] - mc2[0, 0]
    dy_ = mc1[1, 1] - mc1[0, 1]

    print('x0:', x0_, '\n', 'y0:', y0_, '\n', 'dx:', dx_, '\n', 'dy:', dy_)

    m = find_marker.Matching(N_, M_, fps_, x0_, y0_, dx_, dy_)

    start_log = False
    start_time = time.time()
    try:
        while WHILE_COND:
            global reconstru3d, displace_matrix, extend_data_array, T_matrix, T_array, plot_flag

            ret, frame = cap.read()

            if not ret:
                break
            frame = resize_crop_mini(frame, imgw, imgh)
            frame = cv2.flip(frame, 1)

            restruct_img = copy.deepcopy(frame)

            dm = reconstru3d.calc_depth(restruct_img)

            ### find marker masks
            mask = marker_detection.find_marker(frame)

            ### find marker centers
            mc = marker_detection.marker_center(mask, frame)


            if calibrate == False:
                ### matching init
                m.init(mc)
                m.run()

                cur_time = time.time()
                diff_time = cur_time - start_time

                if diff_time > 5:
                    start_log = True

                if SAVE_DATA_FLAG and start_log:
                    flow = m.get_flow()
                    Ox, Oy, Cx, Cy, Occupied = flow
                    displace_matrix = np.zeros((7, 9, 3))

                    max_depth_data = -9999999
                    min_depth_data = 9999999
                    for i in range(len(Ox)):
                        for j in range(len(Ox[i])):
                            depth_data = dm[int(Cy[i][j])][int(Cx[i][j])]
                            displace_matrix[i][j] = [Cx[i][j] - Ox[i][j], Cy[i][j] - Oy[i][j], depth_data]
                            max_depth_data = max(max_depth_data, depth_data)
                            min_depth_data = min(min_depth_data, depth_data)



                    extend_data_array = diagonal_traversal(displace_matrix)

                    T_array = calcForce(extend_data_array)

                    # global plot_flag
                    plot_flag = True

                    T_matrix = restore_Tmatrix(T_array, 7, 9)
                    # print("beforce multiply:", T_matrix[0][0][0], T_matrix[0][0][1], T_matrix[0][0][2])
                    F_matrix = T_matrix * 2.05 * 2.05  # transform to unit N
                    # print("after multiply:", T_matrix[0][0][0], T_matrix[0][0][1], T_matrix[0][0][2])
                    # F_matrix = np.fliplr(F_matrix) # flip vertical
                    draw_flow(frame, flow, F_matrix)

                    flat_F_matrix = F_matrix.reshape(-1, F_matrix.shape[-1])
                    Fx_sum = np.sum(flat_F_matrix[:, 0])
                    Fy_sum = np.sum(flat_F_matrix[:, 1])
                    Fz_sum = np.sum(flat_F_matrix[:, 2])

                    cur_time1 = time.time()
                    diff_time1 = cur_time1 - start_time

                    if force_output:
                        F_list = [[diff_time1, Fx_sum, Fy_sum, Fz_sum, max_depth_data, min_depth_data]]
                        with open(Flist_file, 'a') as file:
                            np.savetxt(file, F_list, delimiter=',', newline='\n', fmt='%f')

                    print(f"Fx_sum: {Fx_sum}, Fy_sum: {Fy_sum}, Fz_sum: {Fz_sum}")

            mask_img = np.asarray(mask)

            bigframe = cv2.resize(frame, (frame.shape[1]*3, frame.shape[0]*3))
            # bigframe = cv2.flip(bigframe, 1)  # 水平翻转180度
            cv2.imshow('frame', bigframe)

            bigmask = cv2.resize(mask_img*255, (mask_img.shape[1]*3, mask_img.shape[0]*3))
            # bigmask = cv2.flip(bigmask, 1)  # 水平翻转180度
            cv2.imshow('mask', bigmask)

            if calibrate:
                cv2.imshow('mask', mask_img*255)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # np.savetxt(file_name, three_dim_array.reshape(-1, three_dim_array.shape[-1]), delimiter=delimiter)
                # print(displace_matrix[0][0], displace_matrix[0][1], displace_matrix[1][0], displace_matrix[2][0])
                cv2.imwrite("cur_img.png", frame)
                np.savetxt("dispalceArray.txt",  extend_data_array, delimiter=', ', fmt='%f')
                np.savetxt("dispalceMatrix.txt", displace_matrix.reshape(-1, displace_matrix.shape[-1]), delimiter=', ', fmt='%f')
                np.savetxt("TMatrix.txt", T_matrix.reshape(-1, T_matrix.shape[-1]), delimiter=', ', fmt='%f')
                np.savetxt("TArray.txt", T_array, delimiter=', ', fmt='%f')
                break
    except KeyboardInterrupt:
        print('Interrupted!')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    heatmap = HeatMap()
    reconstru3d = Reconstruction3D()
    main(sys.argv[1:])

    sys.exit(app.exec_())

