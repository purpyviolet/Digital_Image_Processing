import cv2
from tqdm import tqdm
import subprocess
import os
import cv2
import time  # 确保 time 模块被导入

def video_processing(video_path, set_progress):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('static/output_video.mp4', fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv_frame)
        equalized_v_channel = cv2.equalizeHist(v_channel)
        # equalized_v_channel = cv2.GaussianBlur(equalized_v_channel, (3, 3), 1.5)
        hsv_frame_equalized = cv2.merge([h_channel, s_channel, equalized_v_channel])
        rgb_frame_equalized = cv2.cvtColor(hsv_frame_equalized, cv2.COLOR_HSV2BGR)
        out.write(rgb_frame_equalized)

        frame_count += 1
        progress = (frame_count / total_frames) * 100
        set_progress(progress)
        # # 确保文件路径正确且文件存在
        # if os.path.exists(video_path):
        #     # 在Windows上，可以直接使用 'start' 命令
        #     if os.name == 'nt':
        #         subprocess.run(['start', 'static/output_video.mp4'], shell=True)

    cap.release()
    out.release()


video_path = 'test.mp4'

