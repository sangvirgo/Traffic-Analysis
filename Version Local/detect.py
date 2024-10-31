from datetime import datetime
import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from easyocr import Reader
import cv2
import re
import torch
import pathlib

# Thiết lập các kiểu phương tiện và vi phạm
"""
0: xe may
1: o to
2: xe buyt
3: xe tai
4: doi mu bao hiem
5: khong doi mu bao hiem
6: den do
7: vach ke duong(vach dung)
8: boc dau
9: bien so xe
"""

output_folder = "Image_result"
os.makedirs(output_folder, exist_ok=True)

try:
    pathlib.PosixPath = pathlib.WindowsPath
except Exception as e:
    print(f"Lỗi khi thiết lập pathlib: {e}")

try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
except Exception as e:
    print(f"Lỗi khi tải mô hình YOLOv5: {e}")


def save_violation_bbox(original_image, bbox, name_violation):
    try:
        now = datetime.now()
        violation_id = now.strftime("%H%M%S_%d%m%Y")

        x_min, y_min, x_max, y_max = bbox

        # Cắt ảnh từ bounding box
        cropped_image = original_image.crop((x_min, y_min, x_max, y_max))

        # Đường dẫn file ảnh cho xe vi phạm
        image_folder = os.path.join(output_folder, f"violation_{violation_id}")
        os.makedirs(image_folder, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

        image_path = os.path.join(image_folder, f"{violation_id}.jpg")

        # Lưu ảnh
        cropped_image.save(image_path)
        print(f"Đã phát hiện lỗi {name_violation} của xe vi phạm và lưu vào {image_path}")
    except Exception as e:
        print(f"Lỗi khi lưu ảnh vi phạm: {e}")



def formattedBienSo(bienso):
    return bienso.strip()


def docBienSo(frame):
    try:
        img = cv2.resize(frame, (800, 600))
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        edged = cv2.Canny(blurred, 10, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        number_plate_shape = None
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approximation) == 4:  # rectangle
                number_plate_shape = approximation
                break

        if number_plate_shape is not None:
            (x, y, w, h) = cv2.boundingRect(number_plate_shape)
            number_plate = grayscale[y:y + h, x:x + w]

            try:
                reader = Reader(['en'])
                detection = reader.readtext(number_plate)

                if len(detection) == 0:
                    return None
                else:
                    return formattedBienSo(detection[0][1])
            except Exception as e:
                print(f"Lỗi khi nhận diện biển số: {e}")
                return None
    except Exception as e:
        print(f"Lỗi khi xử lý biển số từ khung hình: {e}")
    return None


def check_vuotdendo(frame, rs):
    try:
        dendo = False
        vachke = None
        vehicles = []
        for det in rs.xyxy[0]:
            if int(det[-1]) == 6:
                dendo = True
            if int(det[-1]) == 7:
                x1, y1, x2, y2 = map(int, det[:4])
                vachke = (y1 + y2) // 2
            if int(det[-1]) in [0, 1, 2, 3]:
                x1, y1, x2, y2 = map(int, det[:4])
                vehicles.append((y1 + y2) // 2)

        if dendo and vachke is not None:
            for v in vehicles:
                if v >= vachke:
                    return True
    except Exception as e:
        print(f"Lỗi khi kiểm tra vượt đèn đỏ: {e}")
    return False


def check_mubh(frame, rs):
    try:
        for det in rs.xyxy[0]:
            if int(det[-1]) == 5:
                return True
    except Exception as e:
        print(f"Lỗi khi kiểm tra không đội mũ bảo hiểm: {e}")
    return False


def bocdau(frame, rs):
    try:
        for det in rs.xyxy[0]:
            if int(det[-1]) == 8:
                return True
    except Exception as e:
        print(f"Lỗi khi kiểm tra bốc đầu: {e}")
    return False


def xuatloi(frame):
    try:
        results = model(frame)

        if check_vuotdendo(frame, results):
            bienso = docBienSo(frame)
            if bienso is not None:
                temp=bienso+" - vuot den do"
                for det in results.xyxy[0]:
                        x1, y1, x2, y2 = map(int, det[:4])
                        save_violation_bbox(Image.fromarray(frame), (x1, y1, x2, y2), temp)
            else:
                for det in results.xyxy[0]:
                        x1, y1, x2, y2 = map(int, det[:4])
                        save_violation_bbox(Image.fromarray(frame), (x1, y1, x2, y2), "vuot den do")

        if check_mubh(frame, results):
            bienso = docBienSo(frame)
            if bienso is not None:
                temp=bienso+" - khong doi mu bao hiem"
                for det in results.xyxy[0]:
                        x1, y1, x2, y2 = map(int, det[:4])
                        save_violation_bbox(Image.fromarray(frame), (x1, y1, x2, y2), temp)
            else:
                for det in results.xyxy[0]:
                        x1, y1, x2, y2 = map(int, det[:4])
                        save_violation_bbox(Image.fromarray(frame), (x1, y1, x2, y2), "khong doi mu bao hiem")

        if bocdau(frame, results):
            bienso = docBienSo(frame)
            if bienso is not None:
                temp=bienso+" - boc dau"
                for det in results.xyxy[0]:
                        x1, y1, x2, y2 = map(int, det[:4])
                        save_violation_bbox(Image.fromarray(frame), (x1, y1, x2, y2), temp)
            else:
                for det in results.xyxy[0]:
                        x1, y1, x2, y2 = map(int, det[:4])
                        save_violation_bbox(Image.fromarray(frame), (x1, y1, x2, y2), "boc dau")

    except Exception as e:
        print(f"Lỗi khi xuất lỗi vi phạm: {e}")
    return frame

try:
    print(model.names)  # Kiểm tra model có hoạt động
except Exception as e:
    print(f"Lỗi khi kiểm tra model.names: {e}")