# 手眼标定（Hand-Eye Calibration）

基于 OpenCV `calibrateHandEye` 的机械臂手眼标定，包含**眼在手上（eye-in-hand）**和**眼在手外（eye-to-hand）**两套完整流程。相机为 Orbbec Gemini 335L，标定板为棋盘格。

## 目录结构

```
.
├── eyeinhand/eyeinhand/          # 眼在手上：相机装在机械臂末端
│   ├── collect_data.py           # 数据采集（Orbbec SDK 取图 + 录入机械臂位姿）
│   ├── main.py                   # 标定主程序
│   ├── RobotToolPose.csv         # 位姿齐次矩阵（main.py 自动生成）
│   └── data/
│       ├── images/               # 棋盘格图片 1.jpg ~ N.jpg
│       └── poses.txt             # 机械臂位姿 x y z rx ry rz
└── eyetohand/eyetohand/          # 眼在手外：相机固定在机械臂外部
    ├── collect_data.py
    ├── main.py
    ├── RobotToolPose.csv
    └── data/
        ├── images/
        └── poses.txt
```

两套流程各自独立，`main.py` 均为单文件、无跨文件依赖，直接运行即可。

## 两者的区别

|  | 眼在手上 eye-in-hand | 眼在手外 eye-to-hand |
|---|---|---|
| 相机安装位置 | 机械臂末端 | 固定在机械臂外部 |
| 标定结果 | 相机相对于**末端**的变换 | 相机相对于**基座**的变换 |
| 位姿矩阵处理 | 直接使用 `base->tool` | 对 `base->tool` **求逆** |

关键差异在 `poses2_main()` 里：眼在手外需要「基座相对于机械臂末端」的变换，即「末端相对于基座」的逆矩阵；眼在手上直接使用不求逆。

## 环境依赖

```
opencv-python
numpy
scipy
pyorbbecsdk    # 仅 collect_data.py 采集数据时需要
```

`main.py` 只需要 opencv / numpy / scipy，不依赖相机 SDK，可以在没有相机的机器上直接跑标定。

## 使用方法

### 1. 采集数据

```bash
python collect_data.py
```

变换机械臂姿态，每个姿态拍一张棋盘格图片并录入对应的机械臂位姿。**建议采集 10~15 组，姿态差异要大**（旋转角度尽量分散，不要只做平移）。

图片按 `1.jpg`、`2.jpg`…… 顺序命名放在 `data/images/`，位姿按相同顺序每行一组写入 `data/poses.txt`。

### 2. 执行标定

```bash
python main.py
```

`main.py` 里的路径配置在文件末尾 `__main__` 块中，按需修改：

```python
images_path = r'...\data\images'
poses_file  = r'...\data\poses.txt'
```

### 3. 输出结果

标定结果以四元数 + 平移向量输出（平移单位为米）：

```
HAND_EYE_QW= 0.61671026
HAND_EYE_QX= -0.35291180
HAND_EYE_QY= 0.60923409
HAND_EYE_QZ= 0.35207321
HAND_EYE_TX=  0.12447500
HAND_EYE_TY=  -0.16936952
HAND_EYE_TZ=  -0.04663608
```

程序会同时跑 PARK / ANDREFF / TSAI 三种算法并打印各自结果，优先选用 PARK 或 ANDREFF 中 `det(R)` 接近 1 的解。**三种方法结果一致且 `det(R) = 1.000000` 说明标定数据质量良好。**

## poses.txt 格式

每行 6 个数：`x y z rx ry rz`，支持逗号或空格分隔，单位自动识别：

```
0.03,0.37,0.65,-0.75,1.47,0.82              # m + rad
-185.290 -261.707 109.540 178.793 4.164 163.819   # mm + deg
```

自动转换规则（`_auto_fix_units`）：

- 平移绝对值中位数 > 5 → 判定为 mm，除以 1000
- 旋转绝对值中位数 > 3.2（≈π）→ 判定为 deg，转成 rad

## 参数调整

**棋盘格参数**在 `hand_eye_calibration()` 的默认参数里：

```python
chessboard_size=(8, 5)   # 内角点数（列, 行），不是方格数
square_size=0.020        # 方格边长，单位米
```

**欧拉角顺序**在 `euler_angles_to_rotation_matrix()`，当前约定为 `R = Rz @ Ry @ Rx`（先绕 z，再绕 y，最后绕 x）。**不同机器人控制器的欧拉角顺序可能不同，如果标定结果方向明显不对，首先检查这里。**

## 结果偏差排查

眼在手外的 `main.py` 中预留了 4 种位姿转换方式（方式 1~4，注释形式），标定结果偏差较大时按顺序尝试：

1. 检查欧拉角顺序是否与机器人控制器一致
2. 切换转换方式（方式 2 / 3 / 4）
3. 打开单位转换 `t_orig = t_orig / 1000.0`
4. 增加更多姿态差异大的图片

有效图片少于 8 张时程序会直接报错，这是刻意的下限保护 —— 图片太少解不可靠。

## 说明

- `imread_chinese()` 用二进制读取 + `cv2.imdecode` 替代 `cv2.imread`，解决中文路径读图失败的问题
- `RobotToolPose.csv` 是 `main.py` 从 `poses.txt` 自动生成的中间产物，格式为 4×(4N) 的横向拼接矩阵，无需手动维护
- 相机 SDK 设备日志（`Log/`）已在 `.gitignore` 中排除
