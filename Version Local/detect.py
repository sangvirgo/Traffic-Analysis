import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from easyocr import Reader
import cv2
import re
import torch
import pathlib

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


pathlib.PosixPath = pathlib.WindowsPath
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)


def save_violation_bbox(original_image, bbox, violation_id):
    """
    Lưu ảnh bounding box của xe vi phạm.

    Parameters:
    - original_image: Ảnh gốc dạng PIL Image
    - bbox: Tọa độ bounding box (x_min, y_min, x_max, y_max)
    - violation_id: ID hoặc số thứ tự của xe vi phạm để tên file ảnh là duy nhất
    """
    x_min, y_min, x_max, y_max = bbox

    # Cắt ảnh từ bounding box
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))

    # Đường dẫn file ảnh cho xe vi phạm
    image_path = os.path.join(output_folder, f"violation_{violation_id}.jpg")

    # Lưu ảnh
    cropped_image.save(image_path)
    print(f"Đã phát hiện lỗi ---- của xe vi phạm và lưu vào {image_path}")





"""Định dạng biển số"""

def save_vipham(bienso, kieuvp):
    with open('vipham.txt', 'a') as logf:
        logf.write(f'Bien so:{bienso}, vi pham: {kieuvp}\n')
def formattedBienSo(bienso):
    return bienso.strip()  # Trả về biển số đã nhận diện


""" doc bien so tu frame """


def docBienSo(frame):
    img = cv2.resize(frame, (800, 600))
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # chuyen sang mau xam cho toi uu
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

        reader = Reader(['en'])
        detection = reader.readtext(number_plate)

        if len(detection) == 0:
            return None  # Không tìm thấy biển số
        else:
            return formattedBienSo(detection[0][1])  # Trả về biển số đã nhận diện
    return None


""" vuot den do """


def check_vuotdendo(frame, rs):
    dendo = False
    vachke = None
    vehicles = []
    for det in rs.xyxy[0]:
        if int(det[-1]) == 6:
            dendo = True
        if int(det[-1]) == 7:
            x1, y1, x2, y2 = map(int, det[:4])
            vachke = (y1 + y2) // 2  # vi hinh anh theo chieu ngang(nen lay y thay vi x)
        if int(det[-1]) in [0, 1, 2, 3]:
            x1, y1, x2, y2 = map(int, det[:4])
            vehicles.append((y1 + y2) // 2)  # toa do trung tam cua cac xe

    if dendo and vachke is not None:
        for v in vehicles:
            if v >= vachke:  # camera thuong o dang sau, nen >=
                return True
    return False


""" ko doi mu bao hiem """


def check_mubh(frame, rs):
    for det in rs.xyxy[0]:
        if int(det[-1]) == 5:
            return True
    return False


""" boc dau"""


def bocdau(frame, rs):
    for det in rs.xyxy[0]:
        if int(det[-1]) == 8:
            return True
    return False


""" xuat ra loi vi pham """


def xuatloi(frame):
    results = model(frame)  # Sử dụng mô hình để dự đoán

    viphams = []
    if check_vuotdendo(frame, results):
        bienso = docBienSo(frame)
        if bienso is not None:
            viphams.append((bienso, "vuot den do"))

    if check_mubh(frame, results):
        bienso = docBienSo(frame)
        if bienso is not None:
            viphams.append((bienso, "khong doi mu bao hiem"))

    if bocdau(frame, results):
        bienso = docBienSo(frame)
        if bienso is not None:
            viphams.append((bienso, "boc dau"))

    for det in results.xyxy[0]:
        clsid = int(det[-1])
        if clsid in [0, 1, 2, 3]:
            x1, y1, x2, y2 = map(int, det[:4])
            midx = (x1 + x2) // 2
            topy = y1

            for bienso, kieuvp in viphams:
                save_vipham(bienso, kieuvp)
                text = f'{bienso} - {kieuvp}'
                cv2.putText(frame, text, (midx, topy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                break

    return frame

# print(model.names)      #kiem tra xem model co hoat dong