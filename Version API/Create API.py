from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__)

# Đường dẫn lưu video upload
UPLOAD_FOLDER = '../uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# API để upload một video kèm tên địa điểm
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files or 'location_name' not in request.form:
        return jsonify({'error': 'Video file and location name are required'}), 400

    video_file = request.files['video']
    location_name = request.form['location_name']

    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Lưu file video với tên địa điểm
    video_filename = f"{location_name}_{video_file.filename}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    video_file.save(video_path)

    return jsonify({'message': 'Video uploaded successfully', 'video': {'location': location_name, 'video_path': video_path}}), 200

# API để lấy video đã upload
@app.route('/video/<filename>', methods=['GET'])
def get_video(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
