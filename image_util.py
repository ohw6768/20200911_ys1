import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def get_face_image(frame, box):
    """
    프레임에서 얼굴이 검출된 영역 계산
    :param frame: 프레임(ndarray)
    :param box: 얼굴 bounding box
    :return: 계산한 영역
    """
    img_height, img_width = frame.shape[:2]
    top, right, bottom, left = box
    box_width = right - left
    box_height = bottom - top

    crop_left = max(left - box_width, 0)
    pad_left = -min(left - box_width, 0)
    crop_top = max(top - box_height, 0)
    pad_top = -min(top - box_height, 0)
    crop_right = min(right + box_width, img_width - 1)
    pad_right = max(right + box_width - img_width, 0)
    crop_bottom = min(bottom + box_height, img_height - 1)
    pad_bottom = max(bottom + box_height - img_height, 0)

    face_image = frame[crop_top:crop_bottom, crop_left:crop_right]
    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return face_image
    padded = cv2.copyMakeBorder(face_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
    return padded


def draw_name(frame, face):
    color = (0, 255, 0)
    thickness = 2
    (top, right, bottom, left) = face.location

    # draw box
    width = 20
    if width > (right - left) // 3:
        width = (right - left) // 3
    height = 20
    if height > (bottom - top) // 3:
        height = (bottom - top) // 3
    cv2.line(frame, (left, top), (left+width, top), color, thickness)
    cv2.line(frame, (right, top), (right-width, top), color, thickness)
    cv2.line(frame, (left, bottom), (left+width, bottom), color, thickness)
    cv2.line(frame, (right, bottom), (right-width, bottom), color, thickness)
    cv2.line(frame, (left, top), (left, top+height), color, thickness)
    cv2.line(frame, (right, top), (right, top+height), color, thickness)
    cv2.line(frame, (left, bottom), (left, bottom-height), color, thickness)
    cv2.line(frame, (right, bottom), (right, bottom-height), color, thickness)

    # draw name
    font = ImageFont.truetype('NanumGothic.ttf', 32)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.text((left+6, bottom+30), face.name, font=font, fill=(255, 255, 255))
    frame = np.array(image)
    return frame

    #cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    # font = cv2.FONT_HERSHEY_DUPLEX
    # cv2.putText(frame, face.name, (left + 6, bottom + 30), font, 1.0,
    #             (255, 255, 255), 1)
