import uuid
from datetime import datetime
import os
from PIL import Image
from easyocr import Reader
import cv2
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
10: nguoi di bo
"""

output_folder = "Image_result"
os.makedirs(output_folder, exist_ok=True)

try:
    """thiet lap duong dan cho pathlib neu khong phai la windows"""
    pathlib.PosixPath = pathlib.WindowsPath
except Exception as e:
    print(f"Lỗi khi thiết lập pathlib: {e}")

try:
    """truyen vao custom model(yolov5) best.pt bang load cuar torch.hub"""
    """ path: duong dan cua may giao vien toi file best.pt them 1 dau \ """
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path = "C:\\Traffic-Analysis\\Main File\\best.pt", force_reload=True) #truyền vào dường dẫn tuyệt đối của file best.pt thay vì dùng best.pt nhuw binhf thuonwgf sex khoong load dduowjc model
except Exception as e:
    print(f"Lỗi khi tải mô hình YOLOv5: {e}")
reader = Reader(['en'], gpu=False)

def formattedBienSo(bienso):
    return bienso.strip()


def docBienSo(frame):
    try:
        img_resized = cv2.resize(frame, (800, 600))
        grayscale = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)

        contrast_enhanced = cv2.convertScaleAbs(grayscale, alpha=1.5, beta=0)

        _, thresh = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        result = reader.readtext(thresh, detail=0)

        bien_so_text = ' '.join(result)
        return bien_so_text.strip() if bien_so_text else None
    except Exception as e:
        print(f"Lỗi khi đọc biển số: {e}")
        return None

def save_violation_bbox(original_image, bbox, name_violation):
    try:
        violation_id = f"{datetime.now().strftime('%H%M%S%f_%d%m%Y')}_{uuid.uuid4()}"
        # vd: 142126282513_03112024_1bba7d07-29d6-4d61-ab5b-0ab8c8746607.jpg
        x_min, y_min, x_max, y_max = bbox

        cropped_image = original_image.crop((x_min, y_min, x_max, y_max))

        image_path = os.path.join(output_folder, f"{violation_id}.jpg")

        cropped_image.save(image_path)
        print(f"Đã phát hiện lỗi {name_violation} của xe vi phạm và lưu vào {image_path}")
    except Exception as e:
        print(f"Lỗi khi lưu ảnh vi phạm: {e}")



"""kiem tra vuot den do"""
def check_vuotdendo(frame, rs):
    try:
        dendo = False
        vachke = None
        vehicles = []
        for det in rs.xyxy[0]:   #duyet tung doi tuong duoc detect [x1, y1, x2, y2, cfd, clsid]
            if int(det[-1]) == 6:   #lay doi tuong tuong ung voi label = 6 (dendo)
                dendo = True
            if int(det[-1]) == 7:
                x1, y1, x2, y2 = map(int, det[:4])      #lay bouding box cua vat the(vach dung)
                vachke = (y1 + y2) // 2            #vi hinh anh la dang chieu ngang nen tinh toan theo y
            if int(det[-1]) in [0, 1, 2, 3]:
                x1, y1, x2, y2 = map(int, det[:4])
                vehicles.append((y1 + y2) // 2)

        if dendo and vachke is not None:
            for v in vehicles:
                if v >= vachke:         #vi camera chieu tu dang sau nen y tang tu tren xuong duoi, nen >= se vi pham
                    return True
    except Exception as e:
        print(f"Lỗi khi kiểm tra vượt đèn đỏ: {e}")
    return False

"""kiem tra khong doi mu bao hiem"""
def check_mubh(frame, rs):
    try:
        for det in rs.xyxy[0]:
            if int(det[-1]) == 5:
                return True
    except Exception as e:
        print(f"Lỗi khi kiểm tra không đội mũ bảo hiểm: {e}")
    return False

"""kiem tra boc dau"""
def bocdau(frame, rs):
    try:
        for det in rs.xyxy[0]:
            if int(det[-1]) == 8:
                return True
    except Exception as e:
        print(f"Lỗi khi kiểm tra bốc đầu: {e}")
    return False


def xuatloi(frame, main_app):
    try:
        results = model(frame)

        if check_vuotdendo(frame, results):
            bienso = docBienSo(frame)
            if bienso is None or bienso is not None:
                bienso = ""
            temp = bienso + " vượt đèn đỏ"
            for det in results.xyxy[0]:
                if int(det[-1]) in [0, 1, 2, 3]:
                    x1, y1, x2, y2 = map(int, det[:4])
                    save_violation_bbox(Image.fromarray(frame), (x1, y1, x2, y2), temp)
                    now = datetime.now()
                    tm = now.strftime("%H:%M:%S")
                    # vd: 20:21:36
                    main_app.log_message(f"Đã phát hiện lỗi{temp} vào lúc {tm}")
                    # Đã phát hiện lỗi vuot den do vào lúc 20:21:36

        if check_mubh(frame, results):
            bienso = docBienSo(frame)
            if bienso is None or bienso is not None:
                bienso = ""
            temp = bienso + " không đội mũ bảo hiểm"
            for det in results.xyxy[0]:
                if int(det[-1]) == 5:
                    x1, y1, x2, y2 = map(int, det[:4])
                    save_violation_bbox(Image.fromarray(frame), (x1, y1, x2, y2), temp)
                    now = datetime.now()
                    tm = now.strftime("%H:%M:%S")
                    main_app.log_message(f"Đã phát hiện lỗi{temp} vào lúc {tm}")

        if bocdau(frame, results):
            bienso = docBienSo(frame)
            if bienso is None or bienso is not None:
                bienso = ""
            temp = bienso + " bốc đầu"
            for det in results.xyxy[0]:
                if int(det[-1]) == 8:
                    x1, y1, x2, y2 = map(int, det[:4])
                    save_violation_bbox(Image.fromarray(frame), (x1, y1, x2, y2), temp)
                    now = datetime.now()
                    tm = now.strftime("%H:%M:%S")
                    main_app.log_message(f"Đã phát hiện lỗi{temp} vào lúc {tm}")

    except Exception as e:
        main_app.log_message(f"Lỗi khi xuất lỗi vi phạm: {e}")
    return frame


try:
    print(model.names)  #kiem thu xem model co hoat dong dung khong voi viec in ra cac class
except Exception as e:
    print(f"Lỗi khi kiểm tra model.names: {e}")