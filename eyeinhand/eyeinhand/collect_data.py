#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from datetime import datetime

import pyorbbecsdk as ob
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode


# ========================= 位姿解析 =========================
class PoseInput:
    @staticmethod
    def parse_xyzuvw(input_str):
        values = [float(x.strip()) for x in input_str.split(',')]
        x, y, z, u, v, w = values

        # mm → m
        x /= 1000.0
        y /= 1000.0
        z /= 1000.0

        R, _ = cv2.Rodrigues(np.radians([u, v, w]))

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T


# ========================= 相机 =========================
class Gemini335LCamera:
    def __init__(self):
        self.pipeline = None
        self.config = None

    def initialize(self):
        self.pipeline = Pipeline()
        self.config = Config()

        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(1280, 720, OBFormat.MJPG, 10)
        self.config.enable_stream(color_profile)

        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_video_stream_profile(848, 480, OBFormat.Y16, 10)
        self.config.enable_stream(depth_profile)

        self.config.set_align_mode(OBAlignMode.SW_MODE)
        self.pipeline.start(self.config)

    def capture_frame(self):
        frames = self.pipeline.wait_for_frames(200)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None

        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img

    def release(self):
        if self.pipeline:
            self.pipeline.stop()


# ========================= 数据采集 =========================
class DataCollector:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.pose_txt_file = os.path.join(output_dir, "poses.txt")

        os.makedirs(self.images_dir, exist_ok=True)

        self.camera = Gemini335LCamera()
        self.pose_input = PoseInput()

    # 保存 txt
    def save_pose_txt(self, raw_str):
        with open(self.pose_txt_file, 'a', encoding='utf-8') as f:
            f.write(raw_str.strip() + "\n")

    # 输入位姿（不会崩）
    def get_user_pose(self):
        while True:
            try:
                input_str = input("\n输入位姿(x,y,z,u,v,w): ").strip()

                if input_str == "":
                    print("⚠️ 输入不能为空")
                    continue

                values = input_str.split(',')
                if len(values) != 6:
                    print("❌ 必须是6个数！")
                    continue

                pose = self.pose_input.parse_xyzuvw(input_str)
                return pose, input_str

            except Exception as e:
                print(f"❌ 输入错误: {e}")
                print("示例: 150,250,400,5,10,15")

    def save_data(self, image, pose, raw_str, index):
        image_filename = f"{index+1}.jpg"
        cv2.imwrite(os.path.join(self.images_dir, image_filename), image)

        # 保存 txt
        self.save_pose_txt(raw_str)

        print(f"✓ 保存成功: {image_filename}  (第 {index+1} 组)")

    def run(self):
        # 清空旧txt
        open(self.pose_txt_file, 'w').close()

        self.camera.initialize()

        index = 0
        win_name = "按 C 采集 | Q 退出"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        while True:
            frame = self.camera.capture_frame()
            if frame is None:
                continue

            # 显示提示信息
            display = frame.copy()
            cv2.putText(display,
                        f"已采集: {index} | C采集 | Q退出",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                captured = frame.copy()

                # 冻结画面（防止卡死）
                cv2.imshow(win_name, captured)
                cv2.waitKey(1)

                print("\n=== 已截图，请输入位姿 ===")

                pose, raw_str = self.get_user_pose()

                self.save_data(captured, pose, raw_str, index)
                index += 1

            elif key == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()


# ========================= main =========================
if __name__ == "__main__":
    collector = DataCollector(
        output_dir=r"C:\Users\HC-YF20250905\Desktop\eye_in_hand_homogeneous_matrix\eyeinhandleft\data"
    )
    collector.run()