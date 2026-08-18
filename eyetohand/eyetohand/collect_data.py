#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gemini 335L 手眼标定数据采集脚本（1280×720）
已确认 D2C 对齐可用
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime

import pyorbbecsdk as ob
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode


class PoseInput:
    """位姿输入处理类"""
    @staticmethod
    def parse_homogeneous_matrix(input_str):
        try:
            rows = input_str.strip().split(';')
            if len(rows) != 4:
                raise ValueError("需要4行数据")
            matrix = [[float(x.strip()) for x in row.split(',')] for row in rows]
            return np.array(matrix, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"矩阵格式错误: {e}")

    @staticmethod
    def parse_xyz_quaternion(input_str):
        try:
            values = [float(x.strip()) for x in input_str.split(',')]
            if len(values) != 7:
                raise ValueError("需要7个值: x,y,z,qx,qy,qz,qw")
            x, y, z, qx, qy, qz, qw = values
            q = np.array([qx, qy, qz, qw]) / np.linalg.norm([qx, qy, qz, qw])
            R = np.array([
                [1 - 2*q[1]**2 - 2*q[2]**2, 2*q[0]*q[1] - 2*q[2]*q[3], 2*q[0]*q[2] + 2*q[1]*q[3]],
                [2*q[0]*q[1] + 2*q[2]*q[3], 1 - 2*q[0]**2 - 2*q[2]**2, 2*q[1]*q[2] - 2*q[0]*q[3]],
                [2*q[0]*q[2] - 2*q[1]*q[3], 2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[0]**2 - 2*q[1]**2]
            ])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            return T
        except Exception as e:
            raise ValueError(f"xyz+四元数格式错误: {e}")

    @staticmethod
    def parse_xyz_rodrigues(input_str):
        try:
            values = [float(x.strip()) for x in input_str.split(',')]
            if len(values) != 6:
                raise ValueError("需要6个值: x,y,z,rx,ry,rz")
            x, y, z, rx, ry, rz = values
            R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            return T
        except Exception as e:
            raise ValueError(f"xyz+旋转向量格式错误: {e}")

    @staticmethod
    def parse_xyzuvw(input_str):
        try:
            values = [float(x.strip()) for x in input_str.split(',')]
            if len(values) != 6:
                raise ValueError("需要6个值: x,y,z,u,v,w")
            x, y, z, u, v, w = values
            x = x / 1000.0
            y = y / 1000.0
            z = z / 1000.0
            R, _ = cv2.Rodrigues(np.radians([u, v, w]))
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            return T
        except Exception as e:
            raise ValueError(f"xyzuvw格式错误: {e}")


class Gemini335LCamera:
    """Gemini 335L 相机控制类（1280×720）"""
    def __init__(self):
        self.pipeline = None
        self.config = None

        # Gemini 335L 1280×720 内参（来自 Orbbec Viewer D2C Camera Param）
        self.camera_matrix = np.array([
            [610.419,   0.0,     639.705],
            [  0.0,   610.307,  368.753],
            [  0.0,     0.0,       1.0 ]
        ], dtype=np.float64)

        # 畸变系数 [k1, k2, p1, p2, k3]（OpenCV 标准顺序）
        # 原始参数: k1=-0.0286534  k2=0.0326785  k3=-0.0116765
        #           p1=-0.000324168  p2=-0.000325691
        self.dist_coeffs = np.array(
            [-0.0286534, 0.0326785, -0.000324168, -0.000325691, -0.0116765],
            dtype=np.float64
        )

    def initialize(self):
        print("正在初始化 Gemini 335L 相机（1280×720 + D2C SW）...")

        self.pipeline = Pipeline()
        self.config = Config()

        # ----------------------------------------------------------------
        # 显式指定流配置，与 Orbbec Viewer 保持一致：
        #   彩色：1280×720  30fps  MJPG
        #   深度：848×480   30fps  Y16
        # 必须先 enable_stream 再 set_align_mode，否则 SDK 校验会失败
        # ----------------------------------------------------------------
        try:
            # 彩色流：1280×720 MJPG 30fps
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_video_stream_profile(
                1280, 720, OBFormat.MJPG, 30
            )
            self.config.enable_stream(color_profile)
            print("彩色流配置: 1280×720 MJPG 30fps")
        except Exception as e:
            print(f"彩色流显式配置失败，使用默认: {e}")
            self.config.enable_stream(OBSensorType.COLOR_SENSOR)

        try:
            # 深度流：848×480 Y16 30fps
            depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_video_stream_profile(
                848, 480, OBFormat.Y16, 30
            )
            self.config.enable_stream(depth_profile)
            print("深度流配置: 848×480 Y16 30fps")
        except Exception as e:
            print(f"深度流显式配置失败，使用默认: {e}")
            self.config.enable_stream(OBSensorType.DEPTH_SENSOR)

        # Gemini 335L 在此分辨率组合下只支持软件 D2C，直接使用 SW_MODE
        try:
            self.config.set_align_mode(OBAlignMode.SW_MODE)
            print("已开启软件 D2C 对齐（SW_MODE）")
        except Exception as e:
            print(f"set_align_mode 失败，将使用默认对齐: {e}")

        self.pipeline.start(self.config)

        # 确认实际分辨率
        try:
            frames = self.pipeline.wait_for_frames(1000)
            if frames:
                color_frame = frames.get_color_frame()
                if color_frame:
                    w = color_frame.get_width()
                    h = color_frame.get_height()
                    print(f"实际采集分辨率: {w} × {h}")
                    if w == 1280 and h == 720:
                        print("✅ 分辨率匹配成功（1280×720）")
                    else:
                        print(f"⚠️  分辨率不匹配，请检查相机配置（期望 1280×720，实际 {w}×{h}）")
        except Exception:
            pass

        print("✅ 相机初始化完成")
        return self.camera_matrix.copy(), self.dist_coeffs.copy()

    def capture_frame(self):
        """返回彩色图像（手眼标定主要使用）"""
        try:
            frames = self.pipeline.wait_for_frames(200)
            if frames is None:
                return None

            color_frame = frames.get_color_frame()
            if color_frame is None:
                return None

            data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

            # MJPG 是压缩格式，必须用 imdecode 解码，不能直接 reshape
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                # 回退：非 MJPG 时按原始 RGB reshape
                h, w = color_frame.get_height(), color_frame.get_width()
                img = data.reshape((h, w, 3))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            print(f"采集失败: {e}")
            return None

    def release(self):
        if self.pipeline:
            self.pipeline.stop()
            print("相机资源已释放")


class DataCollector:
    def __init__(self, output_dir="../data"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.data_file = os.path.join(output_dir, "calibration_data.json")
        os.makedirs(self.images_dir, exist_ok=True)

        self.calibration_data = {
            "camera_matrix": None,
            "dist_coeffs": None,
            "image_files": [],
            "poses": [],
            "timestamp": datetime.now().isoformat(),
            "format": "homogeneous_matrix",
            "camera_model": "Gemini 335L",
            "resolution": "1280x720"
        }

        self.camera = Gemini335LCamera()
        self.pose_input = PoseInput()
        self.format_choice = None
        self.format_type = None

    def select_format(self):
        print("\n" + "="*70)
        print("请选择位姿输入格式（只需选择一次，全程使用该格式）:")
        print("1. 4x4 齐次变换矩阵")
        print("2. xyz + 四元数")
        print("3. xyz + 旋转向量")
        print("4. xyzuvw (xyz单位:mm，uvw单位:度)")
        print("="*70)

        while True:
            choice = input("请输入选择 (1/2/3/4): ").strip()
            if choice in ["1", "2", "3", "4"]:
                self.format_choice = choice
                names = {
                    "1": "homogeneous_matrix",
                    "2": "xyz_quaternion",
                    "3": "xyz_rodrigues",
                    "4": "xyzuvw"
                }
                self.format_type = names[choice]
                self.calibration_data["format"] = self.format_type
                print(f"已固定使用格式: {self.format_type}\n")
                break
            else:
                print("输入错误，请重新选择")

    def get_user_pose(self):
        hints = {
            "1": ("请输入4x4齐次变换矩阵:", "示例: 1,0,0,0.1;0,1,0,0.2;0,0,1,0.3;0,0,0,1"),
            "2": ("请输入xyz+四元数:",       "示例: 0.1,0.2,0.3,0.0,0.0,0.0,1.0"),
            "3": ("请输入xyz+旋转向量:",     "示例: 0.1,0.2,0.3,0.1,0.2,0.3"),
            "4": ("请输入xyzuvw:",           "示例: 150.0,250.0,400.0,5.0,10.0,15.0")
        }
        parsers = {
            "1": self.pose_input.parse_homogeneous_matrix,
            "2": self.pose_input.parse_xyz_quaternion,
            "3": self.pose_input.parse_xyz_rodrigues,
            "4": self.pose_input.parse_xyzuvw,
        }

        title, example = hints[self.format_choice]
        print(f"\n{title}")
        print(example)

        while True:
            try:
                input_str = input("> ").strip()
                pose = parsers[self.format_choice](input_str)
                return pose
            except ValueError as e:
                print(f"输入错误: {e}，请重新输入")

    def save_data(self, image, pose, index):
        image_filename = f"image_{index:04d}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)
        cv2.imwrite(image_path, image)

        self.calibration_data["image_files"].append(image_filename)
        self.calibration_data["poses"].append(pose.tolist())

        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.calibration_data, f, indent=2, ensure_ascii=False)

        print(f"✓ 保存成功: {image_filename}  (第 {index+1} 组)")

    def run(self):
        print("="*80)
        print("Gemini 335L 手眼标定数据采集脚本（1280×720）")
        print("="*80)

        try:
            camera_matrix, dist_coeffs = self.camera.initialize()
            self.calibration_data["camera_matrix"] = camera_matrix.tolist()
            self.calibration_data["dist_coeffs"] = dist_coeffs.tolist()

            self.select_format()

            index = 0
            win_name = "Gemini 335L - 按 C 采集 | 按 Q 退出"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

            while True:
                frame = self.camera.capture_frame()
                if frame is None:
                    continue

                display = frame.copy()
                cv2.putText(display,
                            f"已采集: {index} 组   |   C:采集   Q:退出",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
                cv2.imshow(win_name, display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    captured = frame.copy()
                    pose = self.get_user_pose()
                    self.save_data(captured, pose, index)
                    index += 1

                elif key == ord('q'):
                    print("\n采集结束")
                    break

        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            print(f"\n采集完成！共采集 {len(self.calibration_data['image_files'])} 组数据")
            print(f"数据保存路径: {self.data_file}")


if __name__ == "__main__":
    collector = DataCollector(output_dir=r"C:\Users\HC-YF20250905\Desktop\gemini335l_hand_eye\data")
    collector.run()