import numpy as np
from numpy import linalg
from math import *
#计算旋转矩阵
def get_rotation_matrix(f, w, k):
    R2 = np.zeros((3, 3), dtype=float)
    R2[0, 0] = cos(f) * cos(k) - sin(f) * sin(w) * sin(k)
    R2[0, 1] = -cos(f) * sin(k) - sin(f) * sin(w) * cos(k)
    R2[0, 2] = -sin(f) * cos(w)
    R2[1, 0] = cos(w) * sin(k)
    R2[1, 1] = cos(w) * cos(k)
    R2[1, 2] = -sin(w)
    R2[2, 0] = sin(f) * cos(k) + cos(f) * sin(w) * sin(k)
    R2[2, 1] = -sin(f) * sin(k) + cos(f) * sin(w) * cos(k)
    R2[2, 2] = cos(f) * cos(w)
    return R2
#计算系数矩阵和常数项
def compute_parameters(l, r, df, bx, f, w, k, u, v):
    A = np.zeros((6, 5))
    L = np.zeros(6)
    mr = np.zeros(3)
    for i in range(6):
        x1 = l[i][0]
        y1 = l[i][1]
        z1 = -df
        mr[0] = r[i][0]
        mr[1] = r[i][1]
        mr[2] = -df
        R2 = get_rotation_matrix(f, w, k)
        mrimg = np.dot(R2, mr)
        by = bx * tan(u)
        bz = bx * tan(v)/cos(u)
        N1 = (bx * mrimg[2] - bz * mrimg[0]) / (x1 * mrimg[2] - mrimg[0] * z1)
        N2 = (bx * z1 - bz * x1) / (x1 * mrimg[2] - mrimg[0] * z1)
        Q = N1 * y1 - N2 * mrimg[1] - by
        v1 = np.zeros(6)
        v1[0] = -mrimg[0] * mrimg[1] * N2 / mrimg[2]
        v1[1] = -(mrimg[2] + mrimg[1] * mrimg[1] / mrimg[2]) * N2
        v1[2] = mrimg[0] * N2
        v1[3] = bx
        v1[4] = -mrimg[1] * bx / mrimg[2]
        for ii in range(5):
            A[i][ii] = v1[ii]
        L[i] = Q
    return A, L
#解算五参数
def solve_orientation(A, L):
    X = np.dot(np.dot(linalg.inv(np.dot(A.T, A)), A.T), L)
    return X
#相对定向
def relative_orientation(l, r, df):
    bx = l[0][0] - r[0][0]
    f, w, k, u, v = (0, 0, 0, 0, 0)
    countx = 0
    while True:
        A, L = compute_parameters(l, r, df, bx, f, w, k, u, v)
        X = solve_orientation(A, L)
        f += X[0]
        w += X[1]
        k += X[2]
        u += X[3]
        v += X[4]
        countx += 1
        if (np.abs(X) < 0.00003).all():
            # 计算相对定向精度
            V = np.dot(A, X.T) - L
            c1 = np.sqrt(np.dot(V.T, V) / 6)
            print("相对定向精度：", c1)
            print("相对定向迭代次数：", countx)
            break
    return f, w, k, u, v
#计算地面坐标
def compute_aux_ground_coordinates(projection_coefficients, left_points, right_points, U1V1W1_coordinates, exterior_orientation1, exterior_orientation2):
    num_coords = projection_coefficients.shape[0]
    ground_coordinates = np.zeros((num_coords, 3))
    ground_Y = np.zeros((num_coords, 1))
    for i in range(num_coords):
        N1 = projection_coefficients[i, 0]
        N2 = projection_coefficients[i, 1]
        ground_coordinates[i, 0] = exterior_orientation1[0] + U1V1W1_coordinates[i, 0]
        ground_coordinates[i, 1] = exterior_orientation1[1] + U1V1W1_coordinates[i, 1]
        ground_coordinates[i, 2] = exterior_orientation1[2] + U1V1W1_coordinates[i, 2]
        ground_Y[i, 0] = 0.5 * (exterior_orientation1[1] + N1 * left_points[i, 1] + exterior_orientation2[1] + N2 * right_points[i, 1])
    #print("ground_Y:", ground_Y)
    return ground_coordinates
def aux_ground_coordinates(l, r, df, bx, f, w, k, u, v):
    # 计算地面坐标
    projection_coefficients = np.zeros((l.shape[0], 2))
    for i in range(l.shape[0]):
        x1 = l[i][0]
        y1 = l[i][1]
        z1 = -df
        mr = np.array([r[i][0], r[i][1], -df])
        R2 = get_rotation_matrix(f, w, k)
        mrimg = np.dot(R2, mr)
        by = bx * u
        bz = bx * v
        N1 = (bx * mrimg[2] - bz * mrimg[0]) / (x1 * mrimg[2] - mrimg[0] * z1)
        N2 = (bx * z1 - bz * x1) / (x1 * mrimg[2] - mrimg[0] * z1)
        projection_coefficients[i, 0] = N1
        projection_coefficients[i, 1] = N2

    U1V1W1_coordinates = np.zeros((l.shape[0], 3))
    for i in range(l.shape[0]):
        N1 = projection_coefficients[i, 0]
        U1V1W1_coordinates[i, 0] = N1 * l[i][0]
        U1V1W1_coordinates[i, 1] = N1 * l[i][1]
        U1V1W1_coordinates[i, 2] = N1 * -df

    exterior_orientation1 = np.array([0, 0, 0])  # Placeholder, replace with actual values
    exterior_orientation2 = np.array([0, 0, 0])  # Placeholder, replace with actual values

    ground_coordinates = compute_aux_ground_coordinates(projection_coefficients, l, r, U1V1W1_coordinates, exterior_orientation1, exterior_orientation2)
    return ground_coordinates
def compute_centroid(coords):
    return np.mean(coords, axis=0)

def centralize_coordinates(coords, centroid):
    return coords - centroid

def compute_luvw(centr_model_coords, centr_g_coords,numbda,R,dX,dY,dZ):
    Orient = np.array([[dX], [dY], [dZ]])
    L=(centr_g_coords.T-numbda*np.dot(R,centr_model_coords.T)-Orient).T
    return L
def compute_orientation(A, L):
    return np.dot(np.dot(linalg.inv(np.dot(A.T, A)), A.T), L)

def absolute_orientation(model_points, ground_points):
    # Step 1: 初始化七参数
    phi = omega = kappa = 0
    numbda = 1
    dX = dY = dZ = 0

    # Step 2: 坐标重心化
    model_centroid = compute_centroid(model_points)
    ground_centroid = compute_centroid(ground_points)
    centralized_model_points = centralize_coordinates(model_points, model_centroid)
    centralized_ground_points = centralize_coordinates(ground_points, ground_centroid)
    n=0
    while True:
        n+=1
        print("第",n,"次迭代")
        # step 3: 计算旋转矩阵
        R = get_rotation_matrix(phi, omega, kappa)
        # Step 4: 计算L
        L= compute_luvw(centralized_model_points, centralized_ground_points,numbda,R, dX, dY, dZ).flatten()

        # Step 5: 计算系数阵 A
        A = np.zeros((3 * model_points.shape[0], 7))
        for i in range(model_points.shape[0]):
            U = centralized_model_points[i][0]
            V = centralized_model_points[i][1]
            W = centralized_model_points[i][2]
            Xg = centralized_ground_points[i][0]
            Yg = centralized_ground_points[i][1]
            Zg = centralized_ground_points[i][2]
            A[3 * i]     = [1, 0, 0, U, -W, 0, -V]
            A[3 * i + 1] = [0, 1, 0, V, 0, -W, U]
            A[3 * i + 2] = [0, 0, 1, W, U, V, 0]

        # Step 6: Solve for dphi, domega, dkappa, dnumbda, dX, dY, dZ
        d_params = compute_orientation(A, L)
        dphi, domega, dkappa, dnumbda, ddX, ddY, ddZ = d_params.flatten()

        # Step 7: Update parameters
        numbda *= (1 + dnumbda)
        phi += dphi
        omega += domega
        kappa += dkappa
        dX += ddX
        dY += ddY
        dZ += ddZ
        # Step 8: 计算是否符合精度要求
        if max(abs(dphi), abs(domega), abs(dkappa)) < 0.00003:
            # Step 9: 计算精度
            V = np.dot(A, d_params.T) - L
            c1 = np.sqrt(np.dot(V.T, V) / (3 * model_points.shape[0]))
            print("绝对定向精度：", c1)
            break

    return phi, omega, kappa, numbda, dX, dY, dZ
def calculate_ground_points_from_absolute(numbda, R, dX, dY, dZ, centralized_model_points,ground_centroid):
    Orient = np.array([[dX], [dY], [dZ]])
    ground_points = np.zeros((centralized_model_points.shape[0], 3))
    for i in range(centralized_model_points.shape[0]):
        ground_points[i] = numbda * np.dot(R, centralized_model_points[i].T) + Orient.T + ground_centroid
    return ground_points

if __name__ == "__main__":
    # 左片数据
    l = np.array([[0.016012, 0.079963],
                  [0.08856, 0.081134],
                  [0.013362, -0.07937],
                  [0.08224, -0.080027],
                  [0.051758, 0.080555],
                  [0.014618, -0.000231],
                  [0.04988, -0.000782],
                  [0.08614, -0.001346],
                  [0.048035, -0.079962]])
    # 右片数据
    r = np.array([[-0.07393, 0.078706],
                  [-0.005252, 0.078184],
                  [-0.079122, -0.078879],
                  [-0.009887, -0.080089],
                  [-0.039953, 0.078463],
                  [-0.076006, 0.000036],
                  [-0.042201, -0.001022],
                  [-0.007706, -0.002112],
                  [-0.044438, -0.079736]])
    # l = np.array([[220.0019, 184.0242],
    #               [187.0071, 387.0079],
    #               [ 256.9686, 202.005],
    #               [ 441.0076, 261.0358],
    #               [ 463.0410, 473.0818],
    #               [ 550.9522, 522.0760],
    #               [ 70.9394, 22.4494]])
    # # 右片数据
    # r = np.array([[207.0075, 184.0486],
    #              [190.0083, 387.0102],
    #              [252.0076, 202.0754],
    #              [433.0286, 261.5824],
    #              [470.0779, 473.1367],
    #              [553.0257, 521.3618],
    #              [23.0119, 22.4648]])
    # 控制点左片像点坐标
    lg= np.array([[0.016012, 0.079963],
                  [0.08856, 0.081134],
                  [0.013362, -0.07937],
                  [0.08224, -0.080027]])
    # 控制点右片像点坐标
    rg= np.array([[-0.07393, 0.078706],
                  [-0.005252, 0.078184],
                  [-0.079122, -0.078879],
                  [-0.009887, -0.080089]])
    # 控制点地面摄影测量坐标
    g = np.array([[5083.205, 5852.099, 527.925],
                  [5780.02, 5906.365, 571.549],
                  [5210.879, 4258.446, 461.81],
                  [5909.264, 4314.283, 455.484]])

    # 焦距
    df = 0.15
    #摄影测量比例尺分母
    m=10000
    # 基线长度
    bx=l[0][0]-r[0][0]
    # 相对定向
    orientation = relative_orientation(l, r, df)
    aux_points=m*aux_ground_coordinates(lg, rg, df, bx, *orientation)
    all_aux_points =m*aux_ground_coordinates(l, r, df, bx, *orientation)
    print("控制点空间辅助地面坐标：", aux_points)
    print("相对定向元素:", orientation)
    print("控制点地面测量坐标：", g)
    print("所有空间辅助地面坐标：", all_aux_points)
    # 计算控制点重心
    model_centroid = compute_centroid(aux_points)
    ground_centroid = compute_centroid(g)
    all_model_centroid = compute_centroid(all_aux_points)
    print("模型重心：", model_centroid)
    print("地面重心：", ground_centroid)
    print("所有模型重心：", all_model_centroid)
    # 绝对定向
    phi, omega, kappa, numbda, dX, dY, dZ=absolute_orientation(aux_points, g)
    print("绝对定向参数：", phi, omega, kappa, numbda, dX, dY, dZ)
    all_aux_points=centralize_coordinates(all_aux_points, all_model_centroid)
    # 计算所有点地面坐标
    ground_points = calculate_ground_points_from_absolute(numbda, get_rotation_matrix(phi, omega, kappa), dX, dY, dZ, all_aux_points,ground_centroid)
    print("地面坐标：", ground_points)
