import cv2
import mediapipe as mp
import numpy as np
import math

v = False
l = False
e = False
t = False
o = False


def vec(a, b):
  return a[0] * b[1] - a[1] * b[0]


def get_points(landmark, shape):
  points = []
  for mark in landmark:
    points.append([mark.x * shape[1], mark.y * shape[0]])
  return np.array(points, dtype=np.int32)


def vsize(a):
  return np.linalg.norm(a)


def points(landmark, shape):
  a = get_points(landmark, shape)
  thumb = [a[4][0] - a[1][0], a[4][1] - a[1][1]]
  thumb = np.array(thumb)
  index_finger = [a[8][0] - a[5][0], a[8][1] - a[5][1]]
  index_finger = np.array(index_finger)
  middle_finger = [a[12][0] - a[9][0], a[12][1] - a[9][1]]
  middle_finger = np.array(middle_finger)
  ring_finger = [a[16][0] - a[13][0], a[16][1] - a[13][1]]
  ring_finger = np.array(ring_finger)
  pinky_finger = [a[20][0] - a[17][0], a[20][1] - a[17][1]]
  pinky_finger = np.array(pinky_finger)
  return (thumb, index_finger, middle_finger, ring_finger, pinky_finger)


def V(landmark, shape, flippedRGB):
  global l, e, t, o, v
  p = points(landmark, shape)
  ideal = [0, -100]
  ideal = np.array(ideal)
  if ideal.dot(p[2]) / (vsize(ideal) * vsize(p[2])) < 0.97:
    loc = 0
  elif p[1].dot(p[2]) / (vsize(p[1]) * vsize(p[2])) < 0.98:
    loc = 0
  elif abs(p[2].dot(p[3])) / (vsize(p[2]) * vsize(p[3])) < 0.98:
    loc = 0
  elif abs(p[3].dot(p[4])) / (vsize(p[3]) * vsize(p[4])) < 0.98:
    loc = 0
  else:
    print("В", end="")
    v = True
    l = False
    e = False
    t = False
    o = False


def L(landmark, shape, flippedRGB):
  global l, e, t, o, v
  p = points(landmark, shape)
  ideal = [0, -100]
  ideal = np.array(ideal)
  if ideal.dot(p[2]) / (vsize(ideal) * vsize(p[2])) > -0.97:
    loc = 0
  elif p[2].dot(p[3]) / (vsize(p[2]) * vsize(p[3])) < 0.99:
    loc = 0
  elif abs(p[1].dot(p[2])) / (vsize(p[1]) * vsize(p[2])) < 0.994:
    loc = 0
  elif p[3].dot(p[4]) / (vsize(p[3]) * vsize(p[4])) < 0.98:
    loc = 0
  elif vec(p[1], p[0]) / (vsize(p[1]) * vsize(p[0])) > 0:
    loc = 0
  else:
    print("Л", end="")
    v = False
    l = True
    e = False
    t = False
    o = False


def T(landmark, shape, flippedRGB):
  global l, e, t, o, v
  p = points(landmark, shape)
  ideal = [0, -100]
  ideal = np.array(ideal)
  if ideal.dot(p[2]) / (vsize(ideal) * vsize(p[2])) > -0.97:
    loc = 0
  elif abs(p[1].dot(p[2])) / (vsize(p[1]) * vsize(p[2])) > 0.994:
    loc = 0
  elif abs(p[2].dot(p[3])) / (vsize(p[2]) * vsize(p[3])) > 0.994:
    loc = 0
  elif p[3].dot(p[4]) / (vsize(p[3]) * vsize(p[4])) > 0:
    loc = 0
  elif vec(p[1], p[0]) / (vsize(p[1]) * vsize(p[0])) > 0:
    loc = 0
  else:
    print("Т", end="")
    v = False
    l = False
    e = False
    t = True
    o = False


def E(landmark, shape, flippedRGB):
  a = get_points(landmark, shape)
  global l, e, t, o, v
  p = points(landmark, shape)
  ideal = [0, -100]
  ideal = np.array(ideal)
  sp = [a[6][0] - a[5][0], a[6][1] - a[5][1]]
  sp = np.array(sp)
  id1 = [a[9][0] - a[0][0], a[9][1] - a[0][1]]
  if 0.94 > ideal.dot(id1) / (vsize(ideal) * vsize(id1)) > 0.75:
    loc = 0
  elif sp.dot(p[1]) / (vsize(sp) * vsize(p[1])) > 0.99:
    loc = 0
  elif p[1].dot(p[2]) / (vsize(p[1]) * vsize(p[2])) < 0.99:
    loc = 0
  elif p[2].dot(p[3]) / (vsize(p[2]) * vsize(p[3])) < 0.99:
    loc = 0
  elif p[3].dot(p[4]) / (vsize(p[3]) * vsize(p[4])) < 0.99:
    loc = 0
  else:
    print("Е", end="")
    v = False
    l = False
    e = True
    t = False
    o = False


def O(landmark, shape, flippedRGB):
  a = get_points(landmark, shape)
  global l, e, t, o, v
  p = points(landmark, shape)
  ideal = [0, -100]
  ideal = np.array(ideal)
  sp = [a[6][0] - a[5][0], a[6][1] - a[5][1]]
  sp = np.array(sp)
  if ideal.dot(p[4]) / (vsize(ideal) * vsize(p[4])) < 0.97:
    loc = 0
  elif sp.dot(p[1]) / (vsize(sp) * vsize(p[1])) > 0.99:
    loc = 0
  elif p[1].dot(p[2]) / (vsize(p[1]) * vsize(p[2])) > 0.94:
    loc = 0
  elif p[2].dot(p[3]) / (vsize(p[2]) * vsize(p[3])) < 0.99:
    loc = 0
  elif p[3].dot(p[4]) / (vsize(p[3]) * vsize(p[4])) < 0.99:
    loc = 0
  else:
    print("О", end="")
    v = False
    l = False
    e = False
    t = False
    o = True


#def space(landmark, shape, flippedRGB):
#    a = get_points(landmark, shape)

#создаем детектор
handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
while (cap.isOpened()):
  ret, frame = cap.read()
  if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
    break
  flipped = np.fliplr(frame)
  # переводим его в формат RGB для распознавания
  flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
  # Распознаем
  results = handsDetector.process(flippedRGB)
  # Рисуем распознанное, если распозналось
  if results.multi_hand_landmarks is not None:
    # нас интересует только подушечка указательного пальца (индекс 8)
    # нужно умножить координаты а размеры картинки
    for i in range(21):
      x_tip = int(results.multi_hand_landmarks[0].landmark[i].x *
                  flippedRGB.shape[1])
      y_tip = int(results.multi_hand_landmarks[0].landmark[i].y *
                  flippedRGB.shape[0])
      cv2.circle(flippedRGB, (x_tip, y_tip), 5, (255, 0, 0), -1)

    x_tip1 = int(results.multi_hand_landmarks[0].landmark[9].x *
                 flippedRGB.shape[1])
    y_tip1 = int(results.multi_hand_landmarks[0].landmark[9].y *
                 flippedRGB.shape[0])

    x_tip2 = int(results.multi_hand_landmarks[0].landmark[12].x *
                 flippedRGB.shape[1])
    y_tip2 = int(results.multi_hand_landmarks[0].landmark[12].y *
                 flippedRGB.shape[0])

    shape = flippedRGB.shape
    if v == False:
      V(results.multi_hand_landmarks[0].landmark, shape, flippedRGB)
    if l == False:
      L(results.multi_hand_landmarks[0].landmark, shape, flippedRGB)
    if t == False:
      T(results.multi_hand_landmarks[0].landmark, shape, flippedRGB)
    if e == False:
      E(results.multi_hand_landmarks[0].landmark, shape, flippedRGB)
    if o == False:
      O(results.multi_hand_landmarks[0].landmark, shape, flippedRGB)

  # переводим в BGR и показываем результат
  res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
  cv2.imshow("Hands", res_image)

# освобождаем ресурсы
handsDetector.close()
