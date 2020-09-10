import os
import cv2
import shutil
import face_recognition
import numpy as np
import pickle
from PIL import Image


class Face:
    key = "face_encoding"

    def __init__(self, filename, image, face_encoding, location=None):
        self.filename = filename
        self.image = image  # 얼굴이 검출된 프레임
        self.location = location  # 프레임에서 얼굴 위치
        self.encoding = face_encoding

    def save(self, basedir):
        filepath = os.path.join(basedir, self.filename)
        ext = os.path.splitext(filepath)[1].lstrip('.')
        image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        image.save(filepath, ext)

    @classmethod
    def get_encoding(cls, image):
        rgb = image[:, :, ::-1]
        boxes = face_recognition.face_locations(rgb, model="hog")
        if not boxes:
            height, width, channels = image.shape
            top = int(height / 3)
            bottom = int(top * 2)
            left = int(width / 3)
            right = int(left * 2)
            box = (top, right, bottom, left)
        else:
            box = boxes[0]
        return face_recognition.face_encodings(image, [box])[0]


class Person:
    _last_id = 0

    def __init__(self, name=None):
        if name:
            self.name = name
            if name.startswith('person_') and name[7:].isdigit():
                id = int(name[7:])
                if id > Person._last_id:
                    Person._last_id = id
        else:
            Person._last_id += 1
            self.name = f'person_{Person._last_id}'
        self.encoding = None
        self.faces = []

    @staticmethod
    def descend_last_id():
        Person._last_id -= 1

    def add_face(self, face):
        self.faces.append(face)

    def save_faces(self, basedir):
        filepath = os.path.join(basedir, self.name)
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        for face in self.faces:
            face.save(filepath)

    def set_name(self, name):
        self.name = name
        for face in self.faces:
            face.name = self.name

    def calculate_average_encoding(self):
        if len(self.faces) == 0:
            self.encoding = None
        else:
            encodings = [face.encoding for face in self.faces]
            self.encoding = np.average(encodings, axis=0)

    @classmethod
    def load(cls, filepath, face_encodings):
        basename = os.path.basename(filepath)
        person = Person(basename)
        for face_filename in os.listdir(filepath):
            face_filepath = os.path.join(filepath, face_filename)
            image = cv2.imdecode(np.fromfile(face_filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image.size == 0:
                continue
            if face_filename in face_encodings:
                face_encoding = face_encodings[face_filename]
            else:
                face_encoding = Face.get_encoding(image)
            if face_encoding is not None:
                face = Face(face_filename, image, face_encoding)
                person.faces.append(face)
        person.calculate_average_encoding()
        return person


class PersonDB:
    def __init__(self, unknown_dir='unknown', encoding_file='face_encodings'):
        self.persons = []
        self.unknown_dir = unknown_dir
        self.encoding_file = encoding_file
        self.unknown = Person(self.unknown_dir)
        self.known_name = []

    def add_person(self, person):
        self.persons.append(person)

    def add_name(self, name):
        self.known_name.append(name)

    def load_db(self, dirname):
        if not os.path.isdir(dirname):
            return

        # read face_encodings
        encoding_filepath = os.path.join(dirname, self.encoding_file)
        try:
            with open(encoding_filepath, 'rb') as f:
                face_encodings = pickle.load(f)
        except Exception as e:
            print(f'[ERROR] Cannot pickle face_encodings file by {e}')
            face_encodings = {}
            return

        # read persons
        for entry in os.scandir(dirname):
            if entry.is_dir(follow_symlinks=False):
                filepath = os.path.join(dirname, entry.name)
                person = Person.load(filepath, face_encodings)
                if len(person.faces) == 0:
                    continue
                if entry.name == self.unknown_dir:
                    self.unknown = person
                else:
                    self.add_person(person)

    def save_encodings(self, dirname):
        face_encodings = {}
        for person in self.persons:
            for face in person.faces:
                face_encodings[face.filename] = face.encoding
        for face in self.unknown.faces:
            face_encodings[face.filename] = face.encoding
        filepath = os.path.join(dirname, self.encoding_file)
        with open(filepath, 'wb') as f:
            pickle.dump(face_encodings, f)

    def save_db(self, dirname):
        try:
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            for person in self.persons:
                person.save_faces(dirname)
            self.unknown.save_faces(dirname)
            self.save_encodings(dirname)
        except Exception as e:
            print(f'[ERROR] Cannot save Person DB by {e}')

    def __repr__(self):
        message = f'{len(self.persons)} persons, ' \
                  f'{sum(len(person.faces) for person in self.persons)} known faces, ' \
                  f'{len(self.unknown.faces)} unkonwn faces'
        return message

    def print_persons(self):
        print(self)
