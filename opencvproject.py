import cv2
import mediapipe as mp
import numpy as np
import math

v = False
l = False
e = False
t = False
o = False
spc = False
arr = []
arr1 = []


# Функция, считающее векторное произведение двух вектоов
def vec(a, b):
  return a[0] * b[1] - a[1] * b[0]


# Функция, возвращающая координаты всех точек
def get_points(landmark, shape):
  points = []
  for mark in landmark:
    points.append([mark.x * shape[1], mark.y * shape[0]])
  return np.array(points, dtype=np.int32)


# Функция, считающая длину вектора
def vsize(a):
  return np.linalg.norm(a)


# Функция, создающая вектор для каждого пальца
# по их крайним точкам
def points(landmark, shape):
  a = get_points(landmark, shape)
  # вектор большого пальца
  thumb = [a[4][0] - a[1][0], a[4][1] - a[1][1]]
  thumb = np.array(thumb)
  # вектор указательного пальца
  index_finger = [a[8][0] - a[5][0], a[8][1] - a[5][1]]
  index_finger = np.array(index_finger)
  # вектор среднего пальца
  middle_finger = [a[12][0] - a[9][0], a[12][1] - a[9][1]]
  middle_finger = np.array(middle_finger)
  # вектор безымяного пальца
  ring_finger = [a[16][0] - a[13][0], a[16][1] - a[13][1]]
  ring_finger = np.array(ring_finger)
  # вектор мизинчика
  pinky_finger = [a[20][0] - a[17][0], a[20][1] - a[17][1]]
  pinky_finger = np.array(pinky_finger)

  return (thumb, index_finger, middle_finger, ring_finger, pinky_finger)


# Функция, проверяющая, является ли данный жест
# жестом, обозначающим букву "В"
def V(landmark, shape, flippedRGB):
  # используем эти переменные, чтобы потом понимать, печатали ли мы
  # уже эту букву и не печатать её по несколько раз
  global l, e, t, o, v, spc
  p = points(landmark, shape)
  ideal = [0, -100]
  ideal = np.array(ideal)
  # проверяем, перпендикулярен ли вектор нижней границе по среднему пальцу
  if ideal.dot(p[2]) / (vsize(ideal) * vsize(p[2])) < 0.97:
    loc = 0
  # проверяем, коллинеарны ли вектора указательного и среднего пальцев
  elif p[1].dot(p[2]) / (vsize(p[1]) * vsize(p[2])) < 0.98:
    loc = 0
  # проверяем, коллинеарны ли вектора среднего и безымяного пальцев
  elif abs(p[2].dot(p[3])) / (vsize(p[2]) * vsize(p[3])) < 0.98:
    loc = 0
  # проверяем, коллинеарны ли вектора безымяного пальца и мизинчика
  elif abs(p[3].dot(p[4])) / (vsize(p[3]) * vsize(p[4])) < 0.98:
    loc = 0
  # иначе, выводим букву "В" в консоль и делаем переменную
  # v равной True, дабы показать, что "В" уже выводилась
  else:
    print("В", end="")
    v = True
    l = False
    e = False
    t = False
    o = False
    spc = False


# Функция, проверяющая, является ли данный жест
# жестом, обозначающим букву "Л"
def L(landmark, shape, flippedRGB):
  # используем эти переменные, чтобы потом понимать, печатали ли мы
  # уже эту букву и не печатать её по несколько раз
  global l, e, t, o, v, spc
  p = points(landmark, shape)
  ideal = [0, -100]
  ideal = np.array(ideal)
  # проверяем, перпендикулярен ли вектор верхней границе по среднему пальцу
  if ideal.dot(p[2]) / (vsize(ideal) * vsize(p[2])) > -0.97:
    loc = 0
  # проверяем, коллинеарны ли вектора среднего и безымяного пальцев
  elif p[2].dot(p[3]) / (vsize(p[2]) * vsize(p[3])) < 0.99:
    loc = 0
  # проверяем, коллинеарны ли вектора указательного и среднего пальцев
  elif abs(p[1].dot(p[2])) / (vsize(p[1]) * vsize(p[2])) < 0.994:
    loc = 0
  # проверяем, коллинеарны ли вектора безымяного пальца и мизинчика
  elif p[3].dot(p[4]) / (vsize(p[3]) * vsize(p[4])) < 0.98:
    loc = 0
  # проверяем, с какой стороны находится вектор большого пальца
  # по отношению к указательному
  elif vec(p[1], p[0]) / (vsize(p[1]) * vsize(p[0])) > 0:
    loc = 0
  # иначе, выводим букву "Л" в консоль и делаем переменную
  # l равной True, дабы показать, что "Л" уже выводилась
  else:
    print("Л", end="")
    v = False
    l = True
    e = False
    t = False
    o = False
    spc = False


# Функция, проверяющая, является ли данный жест
# жестом, обозначающим букву "Т"
def T(landmark, shape, flippedRGB):
  # используем эти переменные, чтобы потом понимать, печатали ли мы
  # уже эту букву и не печатать её по несколько раз
  global l, e, t, o, v, spc
  p = points(landmark, shape)
  ideal = [0, -100]
  ideal = np.array(ideal)
  # проверяем, перпендикулярен ли вектор верхней границе по среднему пальцу
  if ideal.dot(p[2]) / (vsize(ideal) * vsize(p[2])) > -0.97:
    loc = 0
  # проверяем, коллинеарны ли вектора указательного и среднего пальцев
  elif abs(p[1].dot(p[2])) / (vsize(p[1]) * vsize(p[2])) > 0.994:
    loc = 0
  # проверяем, коллинеарны ли вектора среднего и безымяного пальцев
  elif abs(p[2].dot(p[3])) / (vsize(p[2]) * vsize(p[3])) > 0.994:
    loc = 0
  # проверяем, коллинеарны ли вектора безымяного пальца и мизинца
  elif p[3].dot(p[4]) / (vsize(p[3]) * vsize(p[4])) > 0:
    loc = 0
  # проверяем, с какой стороны находится вектор большого пальца
  # по отношению к указательному
  elif vec(p[1], p[0]) / (vsize(p[1]) * vsize(p[0])) > 0:
    loc = 0
  # иначе, выводим букву "Т" в консоль и делаем переменную
  # t равной True, дабы показать, что "Т" уже выводилась
  else:
    print("Т", end="")
    v = False
    l = False
    e = False
    t = True
    o = False
    spc = False


# Функция, проверяющая, является ли данный жест
# жестом, обозначающим букву "Е"
def E(landmark, shape, flippedRGB):
  # используем эти переменные, чтобы потом понимать, печатали ли мы
  # уже эту букву и не печатать её по несколько раз
  a = get_points(landmark, shape)
  global l, e, t, o, v, spc
  p = points(landmark, shape)
  ideal = [0, -100]
  ideal = np.array(ideal)
  sp = [a[6][0] - a[5][0], a[6][1] - a[5][1]]
  sp = np.array(sp)
  id1 = [a[9][0] - a[0][0], a[9][1] - a[0][1]]
  # проверяем, перпендикулярен ли вектор нижней границе по среднему пальцу
  if 0.94 > ideal.dot(id1) / (vsize(ideal) * vsize(id1)) > 0.75:
    loc = 0
  # проверяем, коллинеарны ли вектора указателного пальца
  # и его 5-ой и 6-ой точке
  elif sp.dot(p[1]) / (vsize(sp) * vsize(p[1])) > 0.99:
    loc = 0
  # проверяем, коллинеарны ли вектора указательного и среднего пальцев
  elif p[1].dot(p[2]) / (vsize(p[1]) * vsize(p[2])) < 0.99:
    loc = 0
  # проверяем, коллинеарны ли вектора среднего и безымяного пальцев
  elif p[2].dot(p[3]) / (vsize(p[2]) * vsize(p[3])) < 0.99:
    loc = 0
  # проверяем, коллинеарны ли вектора безымяного пальца и мизинца
  elif p[3].dot(p[4]) / (vsize(p[3]) * vsize(p[4])) < 0.99:
    loc = 0
  # иначе, выводим букву "Е" в консоль и делаем переменную
  # e равной True, дабы показать, что "Е" уже выводилась
  else:
    print("Е", end="")
    v = False
    l = False
    e = True
    t = False
    o = False
    spc = False


# Функция, проверяющая, является ли данный жест
# жестом, обозначающим букву "О"
def O(landmark, shape, flippedRGB):
  a = get_points(landmark, shape)
  # используем эти переменные, чтобы потом понимать, печатали ли мы
  # уже эту букву и не печатать её по несколько раз
  global l, e, t, o, v, spc
  p = points(landmark, shape)
  ideal = [0, -100]
  ideal = np.array(ideal)
  sp = [a[6][0] - a[5][0], a[6][1] - a[5][1]]
  sp = np.array(sp)
  # проверяем, перпендикулярен ли вектор нижней границе по среднему пальцу
  if ideal.dot(p[4]) / (vsize(ideal) * vsize(p[4])) < 0.97:
    loc = 0
  # проверяем, коллинеарны ли вектора указателного пальца
  # и его 5-ой и 6-ой точке
  elif sp.dot(p[1]) / (vsize(sp) * vsize(p[1])) > 0.99:
    loc = 0
  # проверяем, коллинеарны ли вектора указательного и среднего пальцев
  elif p[1].dot(p[2]) / (vsize(p[1]) * vsize(p[2])) > 0.94:
    loc = 0
  # проверяем, коллинеарны ли вектора среднего и безымяного пальцев
  elif p[2].dot(p[3]) / (vsize(p[2]) * vsize(p[3])) < 0.99:
    loc = 0
  # проверяем, коллинеарны ли вектора безымяного пальца и мизинца
  elif p[3].dot(p[4]) / (vsize(p[3]) * vsize(p[4])) < 0.99:
    loc = 0
  else:
    print("О", end="")
    v = False
    l = False
    e = False
    t = False
    o = True
    spc = False


# Функция, проверяющая, является ли данный жест
# жестом, обозначающим пробел
def space(landmark, shape, flippedRGB):
  # используем эти переменные, чтобы потом понимать, печатали ли мы
  # уже пробел и не печатать его по несколько раз
  global arr, arr1, spc, l, e, t, o, v
  p = points(landmark, shape)
  a = get_points(landmark, shape)
  b = [a[12][0] - a[8][0], a[12][1] - a[8][1]]
  b = np.array(b)
  if len(arr) < 2:
    arr.append(p)
    arr1.append(b)
  else:
    # проверяем, являлся ли жест два взятия точек назад растопыренной рукой
    if arr[-2][1].dot(arr[-2][2]) / (vsize(arr[-1][1]) * vsize(
        arr[-2][2])) < 0.94 and arr[-2][2].dot(arr[-2][3]) / (
            vsize(arr[-2][2]) * vsize(arr[-2][3])) < 0.94 and arr[-2][3].dot(
                arr[-2][4]) / (vsize(arr[-2][3]) * vsize(arr[-2][4])):
      # проверяем, произошло ли "касание" указательного и среднего пальцев
      if arr1[-2].dot(b) / (vsize(arr1[-2]) * vsize(b)) > 0.99 and p[1].dot(
          p[2]) / (vsize(p[1]) * vsize(p[2])) > 0.94:
        print(" ", end="")
        arr.clear()
        arr1.clear()
        spc = True
        v = False
        l = False
        e = False
        t = False
        o = False
      else:
        arr.append(p)
        arr1.append(b)
    else:
      arr.append(p)
      arr1.append(b)


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
    # нас интересуют все пальцы
    # нужно умножить координаты на размеры картинки
    for i in range(21):
      x_tip = int(results.multi_hand_landmarks[0].landmark[i].x *
                  flippedRGB.shape[1])
      y_tip = int(results.multi_hand_landmarks[0].landmark[i].y *
                  flippedRGB.shape[0])
      # рисуем красные кружочки на координатах всех точек
      cv2.circle(flippedRGB, (x_tip, y_tip), 5, (255, 0, 0), -1)

    # узнаем размеры картинки
    shape = flippedRGB.shape
    # проверяем, если буква "В" до этого не печаталась, вызываем функцию
    if v == False:
      V(results.multi_hand_landmarks[0].landmark, shape, flippedRGB)
    # проверяем, если буква "Л" до этого не печаталась, вызываем функцию
    if l == False:
      L(results.multi_hand_landmarks[0].landmark, shape, flippedRGB)
    # проверяем, если буква "Т" до этого не печаталась, вызываем функцию
    if t == False:
      T(results.multi_hand_landmarks[0].landmark, shape, flippedRGB)
    # проверяем, если буква "Е" до этого не печаталась, вызываем функцию
    if e == False:
      E(results.multi_hand_landmarks[0].landmark, shape, flippedRGB)
    # проверяем, если буква "О" до этого не печаталась, вызываем функцию
    if o == False:
      O(results.multi_hand_landmarks[0].landmark, shape, flippedRGB)
    # проверяем, пробел до этого не печатался, вызываем функцию
    if spc == False:
      space(results.multi_hand_landmarks[0].landmark, shape, flippedRGB)
  # переводим в BGR и показываем результат
  res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
  cv2.imshow("Hands", res_image)

# освобождаем ресурсы
handsDetector.close()
