from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, session
import threading
from HSV import image_processing
from HSV_video import video_processing
import os
import webbrowser  # 导入 webbrowser 模块
from threading import Timer
from find_distance_pure_cv import find_distance_cv
from find_distance import find_distance_dl

progress = 0  # 全局变量来跟踪进度
app = Flask(__name__)
app.secret_key = 'DIP'
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # 上传文件保存的文件夹

def set_progress(value):
    global progress
    progress = value


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    # 保存上传的文件
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    # 处理图片
    image_processing(filename)  # 调用 HSV.py 中的处理函数，处理图片并保存在 static 文件夹中

    return redirect(url_for('result'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    video = request.files['video']
    if video.filename == '':
        return redirect(url_for('index'))

    filename = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(filename)

    # 使用线程处理视频，并等待线程完成
    thread = threading.Thread(target=video_processing, args=(filename, set_progress))
    thread.start()
    thread.join()  # 这里我们等待处理线程结束

    return redirect(url_for('result_video'))

@app.route('/result')
def result():
    result_files = [
        'original_image.jpg',
        'hsv_image.jpg',
        'h_image.jpg',
        's_image.jpg',
        'v_image.jpg',
        'equalized_v_image.jpg',
        'equalized_rgb_image.jpg'
    ]
    result_paths = [os.path.join('static', f) for f in result_files]

    return render_template('result.html', result_paths=result_paths)


@app.route('/result_video')
def result_video():
    video_path = url_for('static', filename='output_video.mp4')
    return render_template('result_video.html', video_path=video_path)

@app.route('/static/<filename>')
def custom_static(filename):
    return send_from_directory('static', filename)


@app.route('/progress')
def get_progress():
    return jsonify({'progress': progress})


@app.route('/upload_distance', methods=['POST'])
def upload_distance():
    if 'distance_file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['distance_file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    # 调用 find_distance_cv 函数
    find_distance_cv(filename)

    return redirect(url_for('result_distance'))


@app.route('/result_distance')
def result_distance():
    distance_path = url_for('static', filename='find_distance/distance_pic.jpg')
    edged_path = url_for('static', filename='find_distance/edged_pic.jpg')
    return render_template('distance_cv.html', distance_path=distance_path, edged_path=edged_path)


@app.route('/upload_distance_ml', methods=['POST'])
def upload_distance_ml():
    if 'distance_file_ml' not in request.files:
        return redirect(url_for('index'))

    file = request.files['distance_file_ml']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    # 调用 find_distance_dl 函数
    output = find_distance_dl(filename)
    # print(output)

    # 将输出信息存储在会话中
    session['output'] = output

    return redirect(url_for('result_distance_ml'))


@app.route('/result_distance_ml')
def result_distance_ml():
    # 从会话中获取输出信息
    output = session.get('output', None)
    if output is None:
        return redirect(url_for('index'))

    distance_path_ml = url_for('static', filename='find_distance/distance_dl.jpg')
    return render_template('result_distance_ml.html', distance_path=distance_path_ml, output=output)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')


if __name__ == '__main__':
    # 使用 Timer 延迟几秒钟打开网页，以确保 Flask 服务器已经启动
    Timer(1, open_browser).start()
    app.run(debug=True)
