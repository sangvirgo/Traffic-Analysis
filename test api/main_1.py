import requests
from detect import xuatloi

def download_video(url, local_filename):
    # Gửi request để tải video
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Lưu video vào file local
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video downloaded successfully: {local_filename}")
        return local_filename
    else:
        print("Failed to download video")
        return None


def main():
    # URL video từ API
    video_url = 'http://127.0.0.1:5000/video/Le Van Viet.mp4'
    
    # Đường dẫn lưu video cục bộ
    local_video_path = 'downloaded_video.mp4'

    # Tải video
    downloaded_video = download_video(video_url, local_video_path)
    if downloaded_video:
        # Xử lý video
        xuatloi(downloaded_video)

if __name__ == '__main__':
    main()