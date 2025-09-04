import cv2
import numpy as np

# グローバル変数の初期化
zoom_factor = 1.0
pan_x, pan_y = 0, 0


def update_zoom(image: np.ndarray):
    global zoom_factor, pan_x, pan_y
    h, w = image.shape[:2]

    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)

    max_pan_x = (w - new_w) // 2
    max_pan_y = (h - new_h) // 2
    pan_x = max(-max_pan_x, min(pan_x, max_pan_x))
    pan_y = max(-max_pan_y, min(pan_y, max_pan_y))

    center_x, center_y = w // 2 + pan_x, h // 2 + pan_y
    x1 = max(center_x - new_w // 2, 0)
    y1 = max(center_y - new_h // 2, 0)
    x2 = min(center_x + new_w // 2, w)
    y2 = min(center_y + new_h // 2, h)

    cropped = image[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed


def on_key(key: int, image: np.ndarray):
    global zoom_factor, pan_x, pan_y
    h, w = image.shape[:2]

    if (key & 0xFF) == ord("+"):
        zoom_factor *= 1.1
    elif (key & 0xFF) == ord("-"):
        zoom_factor /= 1.1
    elif (key & 0xFF) == ord("8"):
        pan_y -= 20
    elif (key & 0xFF) == ord("2"):
        pan_y += 20
    elif (key & 0xFF) == ord("4"):
        pan_x -= 20
    elif (key & 0xFF) == ord("6"):
        pan_x += 20

    # ズーム倍率とパン位置を制限
    zoom_factor = max(1.0, min(zoom_factor, 10.0))
    zoomed = update_zoom(image)
    cv2.imshow("Zoomed Image", zoomed)


def main(image_path: str):
    global zoom_factor, pan_x, pan_y

    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return

    cv2.imshow("Zoomed Image", image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESCキーで終了
            break
        elif key in [ord("+"), ord("-"), ord("8"), ord("2"), ord("4"), ord("6")]:
            on_key(key, image)

    cv2.destroyAllWindows()


# 使用例
main("testpy/IMG_8678.JPG")
