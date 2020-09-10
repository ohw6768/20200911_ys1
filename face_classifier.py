import face_recognition
from datetime import datetime
from collections import Counter

from person_db import Face
from person_db import Person
from person_db import PersonDB
from image_util import *


class FaceClassifier:
    def __init__(self, threshold, ratio):
        self.threshold = threshold
        self.ratio = ratio

    def locate_faces(self, frame):
        """
        frame에서 얼굴이 위치한 bounding box 검출
        :param frame: 프레임(ndarray)
        :return: bounding box list
        """
        if self.ratio == 1.0:
            rgb = frame[:, :, ::-1]
        else:
            resize_frame = cv2.resize(frame, (0, 0), fx=self.ratio, fy=self.ratio)
            rgb = resize_frame[:, :, ::-1]

        boxes = face_recognition.face_locations(rgb)
        if self.ratio == 1.0:
            return boxes

        new_boxes = []
        for box in boxes:
            top, right, bottom, left = box
            left = int(left / ratio)
            top = int(top / ratio)
            right = int(right / ratio)
            bottom = int(bottom / ratio)
            new_boxes.append((top, right, bottom, left))
        return new_boxes

    def detect_faces(self, frame):
        """
        frame에서 얼굴 검출
        :param frame: 프레임(ndarray)
        :return: 검출한 얼굴로 생성한 Face 객체 리스트
        """
        boxes = self.locate_faces(frame)  # 프레임에서 얼굴 위치 검출
        if len(boxes) == 0:
            return []

        faces = []
        prefix = datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
        encodings = face_recognition.face_encodings(frame, boxes)
        for i, box in enumerate(boxes):
            face_image = get_face_image(frame, box)
            face = Face(f'{prefix}_{i}.png', face_image, encodings[i], location=box)
            faces.append(face)
        return faces

    def compare_with_known_persons(self, face, persons):
        """
        검출한 얼굴이 기존에 인식한 사람인지 비교하고,
        기존에 인식한 사람이면 Person 객체 반환
        아니면 None 반환
        :param face: 검출한 얼굴
        :param persons: 인식한 사람 목록
        :return: Person 객체 또는 None
        """
        if len(persons) == 0:
            return None

        encodings = [person.encoding for person in persons]
        distances = face_recognition.face_distance(encodings, face.encoding)
        index = np.argmin(distances)
        min_value = distances[index]
        if min_value < self.threshold:
            persons[index].add_face(face)
            persons[index].calculate_average_encoding()
            face.name = persons[index].name
            return persons[index]
        else:
            return None

    def compare_with_unknown_persons(self, face, unknown_faces):
        """
        검출한 얼굴이 기존에 미인식된 사람인지 비교하고,
        기존에 미인식된 사람이면 새로운 Person 객체 생성
        아니면 None 반환
        :param face: 검출한 얼굴
        :param unknown_faces: 미인식한 사람 목록
        :return: Person 객체 또는 None
        """
        if len(unknown_faces) == 0:
            face.name = "unknown"
            unknown_faces.append(face)
            return None

        encodings = [face.encoding for face in unknown_faces]
        distances = face_recognition.face_distance(encodings, face.encoding)
        index = np.argmin(distances)
        min_value = distances[index]
        if min_value < self.threshold:
            person = Person()
            newly_known_face = unknown_faces.pop(index)
            person.add_face(newly_known_face)
            person.add_face(face)
            person.calculate_average_encoding()
            face.name = person.name
            newly_known_face.name = person.name
            return person
        else:
            face.name = "unknown"
            unknown_faces.append(face)
            return None


if __name__ == '__main__':
    import argparse
    import signal
    import os

    args = argparse.ArgumentParser()
    # 동영상 파일 경로. 웹캠에서 프레임을 받아올 경우 0으로 지정
    args.add_argument("inputfile",
                      help="video file to detect or '0' to detect from web cam")
    # 인식된 얼굴간의 유사도를 비교할 기준.
    # 값이 높을 경우 서로 다른 사람을 같은 사람으로 인식할 수 있음
    args.add_argument("-t", "--threshold", default=0.44, type=float,
                      help="threshold of the similarity (default=0.44)")
    args.add_argument("-S", "--seconds", default=1, type=float,
                      help="seconds between capture")
    args.add_argument("-s", "--stop", default=0, type=int,
                      help="stop detecting after # seconds")
    args.add_argument("-k", "--skip", default=0, type=int,
                      help="skip detecting for # seconds from the start")
    args.add_argument("-d", "--display", action='store_true',
                      help="display the frame in real time")
    args.add_argument("-c", "--capture", type=str,
                      help="save the frames with face in the CAPTURE directory")
    # 프레임을 리사이즈할 비율. 작게 리사이즈할 경우 분석 시간이 짧지만 정확도가 낮아질 수 있음
    args.add_argument("-r", "--resize-ratio", default=1.0, type=str,
                      help="resize the frame to process (less time, less accuracy)")
    # 인식할 사람 이름 목록이 있는 텍스트 파일 경로.
    args.add_argument("-n", "--name", default="", type=str,
                      help="people name to want detect.")
    _args = args.parse_args()

    if _args.stop < _args.skip:
        print('[ERROR] Stop parameter must be larger than Skip parameter.')
        exit(1)

    if _args.capture and not os.path.isdir(_args.capture):
        try:
            os.mkdir(_args.capture)
        except Exception as e:
            print(f'[ERROR] Cannot make dir at {_args.capture} by {e}')
            exit(1)

    name_list = []
    if _args.name:
        with open(_args.name, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                name_list.append(line.strip())

    def is_all_person_detected(all_name_list, detected_name_list):
        return Counter(all_name_list) == Counter(detected_name_list)

    # SIGINT(^C|ctrl+c) handler
    def signal_handler(sig, frame):
        global running
        running = False
    prev_handler = signal.signal(signal.SIGINT, signal_handler)

    src = _args.inputfile
    if src == '0':
        print('[INFO] Detect from web cam')

    video = cv2.VideoCapture(src)
    if not video.isOpened():
        if src == '0':
            print('[ERROR] Cannot get frame from the web cam.')
        else:
            print(f'[ERROR] Cannot get frame from the {src}')
        exit(1)

    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # 프레임 가로 크기
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 프레임 세로 크기
    frame_rate = video.get(cv2.CAP_PROP_FPS)  # fps
    capture_interval = int(round(frame_rate * _args.seconds))  # 캡쳐할 프레임 단위

    print("============================================================")
    print('Web Cam') if src == '0' else print(src)
    print(f'{frame_width}x{frame_height}, {frame_rate} fps')
    if _args.resize_ratio != 1.0:
        ratio = _args.resize_ratio
        resize_width = int(frame_width * ratio)
        resize_height = int(frame_height * ratio)
        print(f'-> {resize_width}x{resize_height}')
    print(f'Capture every {capture_interval} frame.')
    print(f'Face similarity threshold: {_args.threshold}')
    print("============================================================")
    if _args.display:
        print('Press Q to stop detecting...')

    result_dir = "result"
    pdb = PersonDB()
    pdb.load_db(result_dir)
    pdb.print_persons()

    fc = FaceClassifier(_args.threshold, _args.resize_ratio)
    frame_idx = 0
    running = True

    while running:
        ret, frame = video.read()
        if frame is None:
            print('[INFO] Frame is None.')
            break

        frame_idx += 1
        if frame_idx % capture_interval != 0:
            continue

        seconds = frame_idx / frame_rate
        if 0 < _args.stop < seconds:
            break
        if seconds < _args.skip:
            continue

        faces = fc.detect_faces(frame)
        for face in faces:
            person = fc.compare_with_known_persons(face, pdb.persons)
            if person:
                continue
            person = fc.compare_with_unknown_persons(face, pdb.unknown.faces)
            if person:
                if not is_all_person_detected(name_list, pdb.known_name):
                    print('Choose person name. Insert index number or P to pass.')
                    print('index. name')
                    for i, name in enumerate(name_list):
                        print(f'{i:5}. {name}')
                    selected_index = input('--> ')
                    if selected_index.lower() != 'p':
                        while not selected_index.isdecimal() or int(selected_index) >= len(name_list):
                            print('Insert index number.')
                            selected_index = input('--> ')
                        print(f'You choose {selected_index}. {name_list[int(selected_index)]}.')
                        person.set_name(name_list[int(selected_index)])
                        person.descend_last_id()
                        pdb.add_name(person.name)
                pdb.add_person(person)

        if _args.display or _args.capture:
            for face in faces:
                frame = draw_name(frame, face)

        if _args.display:
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False

        if _args.capture and len(faces) > 0:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]}.png"
            filepath = os.path.join(_args.capture, filename)
            cv2.imwrite(filepath, frame)

    signal.signal(signal.SIGINT, prev_handler)
    running = False
    video.release()

    pdb.save_db(result_dir)
    # pdb.print_persons()
    print('Detected person list : ')
    for name in pdb.known_name:
        print(name)
