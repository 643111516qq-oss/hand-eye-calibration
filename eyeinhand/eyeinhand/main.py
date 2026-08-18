# coding=utf-8
import os
import re
import csv
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=8, suppress=True)


def imread_chinese(path):
    """支持中文路径的图片读取"""
    try:
        # 使用 numpy 从文件读取二进制数据
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        # 解码图片
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"读取失败: {path}, 错误: {e}")
        return None



# ================== 位姿处理函数 ==================

def poses2_main(pose_path: str):
    
    # 1) 读取并解析 pose 文件
    poses = _read_poses(pose_path)  # shape (N,6)

    # 2) 单位自动修正（mm->m, deg->rad）
    poses = _auto_fix_units(poses)

    # 3) 逐条 pose 转齐次矩阵（眼在手上直接使用，不需要求逆）
    matrices = []
    for p in poses:
        H_tool_in_base = pose_to_homogeneous_matrix(p)
        matrices.append(H_tool_in_base)

    # 4) 输出到脚本目录
    out_csv = os.path.join(os.path.dirname(__file__), "RobotToolPose.csv")
    save_matrices_to_csv(matrices, out_csv)

    print(f"[poses2] Loaded poses: {len(poses)}")
    print(f"[poses2] Saved: {out_csv}")


def _read_poses(pose_path: str) -> np.ndarray:
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"pose file not found: {pose_path}")

    with open(pose_path, "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    rows = []
    for ln in raw_lines:
        parts = re.split(r"[,\s]+", ln)
        if len(parts) < 6:
            continue
        try:
            vals = [float(x) for x in parts[:6]]
        except ValueError:
            continue
        rows.append(vals)

    if len(rows) == 0:
        raise RuntimeError(f"No valid pose rows parsed from: {pose_path}")

    return np.array(rows, dtype=np.float64)


def _auto_fix_units(poses: np.ndarray) -> np.ndarray:
    poses = poses.copy()

    # 平移：如果中位数绝对值 > 5，基本可以判断是 mm，转成 m
    if np.median(np.abs(poses[:, :3])) > 5.0:
        poses[:, :3] /= 1000.0

    # 旋转：如果中位数绝对值 > 3.2，基本可以判断是 deg，转成 rad
    if np.median(np.abs(poses[:, 3:])) > 3.2:
        poses[:, 3:] = np.deg2rad(poses[:, 3:])

    return poses


def euler_angles_to_rotation_matrix(rx, ry, rz):
    """
    输入：rx,ry,rz (弧度)
    约定：R = Rz @ Ry @ Rx  （先绕z，再绕y，再绕x）
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]], dtype=np.float64)

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float64)

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


def pose_to_homogeneous_matrix(pose):
    """
    pose: [x, y, z, rx, ry, rz]
    单位：x,y,z 米；rx,ry,rz 弧度
    """
    x, y, z, rx, ry, rz = pose
    Rm = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z], dtype=np.float64).reshape(3, 1)

    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = Rm
    H[:3, 3] = t[:, 0]
    return H


def save_matrices_to_csv(matrices, file_name):
    if len(matrices) == 0:
        raise RuntimeError("No matrices to save.")

    rows, cols = matrices[0].shape
    num_matrices = len(matrices)

    combined = np.zeros((rows, cols * num_matrices), dtype=np.float64)
    for i, mat in enumerate(matrices):
        combined[:, i * cols:(i + 1) * cols] = mat

    with open(file_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in combined:
            writer.writerow(row.tolist())


# ================== 手眼标定函数 ==================

def hand_eye_calibration(images_path, poses_file, chessboard_size=(8, 5), square_size=0.020):
    """
    执行眼在手上手眼标定

    参数:
        images_path: 标定板图片路径
        poses_file: 机械臂位姿文件路径
        chessboard_size: 棋盘格尺寸 (列数, 行数)
        square_size: 棋盘格边长 (米)

    返回:
        rotation_matrix, translation_vector
    """
    XX, YY = chessboard_size

    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2) * square_size

    obj_points = []
    img_points = []
    size = None

    print("正在检测棋盘格...")
    for i in range(1, 50):
        img_path = os.path.join(images_path, f"{i}.jpg")
        if not os.path.exists(img_path):
            continue
        img = imread_chinese(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
        if not ret:
            continue

        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)

    N = len(img_points)
    if N < 8:
        raise RuntimeError(f"有效图片太少，仅检测到 {N} 张，请确保至少 8-15 张姿态差异大的图片")

    print(f"成功检测到 {N} 张有效图片")

    # 相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print("\n相机内参矩阵:\n", mtx)
    print("畸变系数:\n", dist)

    # ================== 读取机械臂位姿 ==================
    poses2_main(poses_file)
    tool_pose = np.loadtxt(f'{os.path.dirname(__file__)}/RobotToolPose.csv', delimiter=',')

    R_tool = []
    t_tool = []

    for i in range(N):
        # 眼在手上：直接使用 base->tool 的变换
        R_g = tool_pose[0:3, 4*i:4*i+3].copy()
        t_g = tool_pose[0:3, 4*i+3].copy()

        R_tool.append(R_g)
        t_tool.append(t_g)

    print("\n已加载 Tool 位姿（眼在手上模式）")

    # ====================== 手眼标定 ======================
    # 眼在手上标定方程：A2^{-1}*A1*X = B2*B1^{-1}*X
    # 其中 X 是相机相对于末端的变换 (camera -> gripper)
    methods = {
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    }

    best_R = None
    best_t = None
    best_method = ""

    for name, flag in methods.items():
        # 眼在手上：标定板相对于相机的变换 (target -> camera)
        # 注意：这里需要将 target2cam 传入
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_tool, t_tool, rvecs, tvecs, flag
        )
        det = np.linalg.det(R_cam2gripper)
        t_flat = t_cam2gripper.flatten()

        print(f"\n--- 方法: {name} ---")
        print("旋转矩阵 R (camera -> gripper):\n", R_cam2gripper)
        print("平移向量 t (米):", t_flat)
        print(f"det(R) = {det:.6f}")

        if name in ["PARK", "ANDREFF"] and abs(det - 1.0) < 0.05:
            best_R = R_cam2gripper
            best_t = t_cam2gripper
            best_method = name

    if best_R is None:
        best_R, best_t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, cv2.CALIB_HAND_EYE_PARK)
        best_method = "PARK (fallback)"

    print(f"\n最终选用方法: {best_method}")

    return best_R, best_t


# ====================== 主程序 ======================
if __name__ == '__main__':
    # 配置路径
    images_path = r'D:\研二下\biaoding\eyeinhand\eyeinhand\data\images'
    poses_file = r'D:\研二下\biaoding\eyeinhand\eyeinhand\data\poses.txt'

    # 执行手眼标定
    rotation_matrix, translation_vector = hand_eye_calibration(images_path, poses_file)

    # 转换为四元数 (scipy 返回 xyzw)
    rot = R.from_matrix(rotation_matrix)
    quat = rot.as_quat()
    qx, qy, qz, qw = quat

    x, y, z = translation_vector.flatten()

    print("\n=== 最终手眼标定结果（眼在手上：相机相对于末端） ===")
    print(f"HAND_EYE_QW= {qw:.8f}")
    print(f"HAND_EYE_QX= {qx:.8f}")
    print(f"HAND_EYE_QY= {qy:.8f}")
    print(f"HAND_EYE_QZ= {qz:.8f}")
    print(f"HAND_EYE_TX=  {x:.8f}")
    print(f"HAND_EYE_TY=  {y:.8f}")
    print(f"HAND_EYE_TZ=  {z:.8f}")
