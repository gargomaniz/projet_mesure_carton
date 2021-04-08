import cv2
import matplotlib.pyplot as plt

def find_maxarea(contours):
  max_area = cv2.contourArea(contours[0])
  index_max_area = 0
  for i, contour in enumerate(contours):
    if cv2.contourArea(contours[i])> max_area:
      max_area = cv2.contourArea(contours[i])
      index_max_area = i
  return i


def segment_points(mask):
  ret, thresh = cv2.threshold(mask, 100, 255, 0)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  mask2_real = np.zeros(mask.shape, np.uint8)

  index_max_area = find_maxarea(contours)

  cnts=[]
  cnts_poly = cv2.approxPolyDP(contours[index_max_area], 5, True)

  cnts.append(cnts_poly)

  cv2.drawContours(mask2_real, cnts, 0, 255)

  cnts_poly = cnts_poly.reshape(len(cnts_poly), 2)
  return cnts_poly

import xml.etree.ElementTree as ET
import numpy as np

def load_from_opencv_xml(filename, elementname, dtype='float32'):
    try:
        tree = ET.parse(filename)
        rows = int(tree.find(elementname).find('rows').text)
        cols = int(tree.find(elementname).find('cols').text)
        return np.fromstring(tree.find(elementname).find('data').text, dtype, count=rows*cols, sep=' ').reshape((rows, cols))
    except Exception as e:
        print(e)
        return None


def select_good_points_pos1(cnts_poly_e):
  cnts_poly = cnts_poly_e
  MaxY = cnts_poly[0][1]
  MaxX = cnts_poly[0][0]
  MinY = cnts_poly[0][1]
  MinX = cnts_poly[0][0]

  g_points = []

  for i in range(len(cnts_poly)):
    if cnts_poly[i][1] >= MaxY:
      MaxY = cnts_poly[i][1]
    if (cnts_poly[i][1] <= MinY):
      MinY = cnts_poly[i][1]
    if (cnts_poly[i][0] >= MaxX):
      MaxX = cnts_poly[i][0]
    if (cnts_poly[i][0] <= MinX):
      MinX = cnts_poly[i][0]

  indexes = [91, 91, 91, 91, 91, 91, 91, 91]

  # recup points en bas à gauche -> 1
  # laisser en bas à droite à 0 _2

  # récup en haut à gauche -3
  # éventuellement en haut au milieu si existe -4

  # recup en haut à droite -6
  # laisser en haut opposéL à 0 -5

  # 8 est en bas à droite
  # laisser 7 à 0

  for i in range(len(cnts_poly)):
    if abs(cnts_poly[i][0] - MinX < 15):
      miny_local = cnts_poly[i][1]
      maxy_local = cnts_poly[i][1]
      for j in range(len(cnts_poly)):
        if ((abs(cnts_poly[j][0] - MinX) < 15) and (cnts_poly[j][1] <= miny_local)):
          miny_local = cnts_poly[j][1]
          indexes[2] = j
        if ((abs(cnts_poly[j][0] - MinX) < 15) and (cnts_poly[j][1] >= maxy_local)):
          maxy_local = cnts_poly[j][1]
          indexes[0] = j
    if abs(cnts_poly[i][0] - MaxX < 15):
      miny_local = cnts_poly[i][1]
      maxy_local = cnts_poly[i][1]
      for j in range(len(cnts_poly)):
        if ((abs(cnts_poly[j][0] - MaxX) < 20) and (cnts_poly[j][1] <= miny_local)):
          miny_local = cnts_poly[j][1]
          indexes[5] = j
        if ((abs(cnts_poly[j][0] - MaxX) < 20) and (cnts_poly[j][1] >= maxy_local)):
          maxy_local = cnts_poly[j][1]
          indexes[7] = j

  g_points = []

  for i, index in enumerate(indexes):
    if (indexes[i] != 91):
      g_points.append(cnts_poly[index])
    else:
      g_points.append(np.array([0, 0]))

  return g_points


def select_good_points_pos2(cnts_poly_e):
  cnts_poly = cnts_poly_e
  MaxY = cnts_poly[0][1]
  MaxX = cnts_poly[0][0]
  MinY = cnts_poly[0][1]
  MinX = cnts_poly[0][0]

  g_points = []

  for i in range(len(cnts_poly)):
    if cnts_poly[i][1] >= MaxY:
      MaxY = cnts_poly[i][1]
    if (cnts_poly[i][1] <= MinY):
      MinY = cnts_poly[i][1]
    if (cnts_poly[i][0] >= MaxX):
      MaxX = cnts_poly[i][0]
    if (cnts_poly[i][0] <= MinX):
      MinX = cnts_poly[i][0]

  indexes = [91, 91, 91, 91, 91, 91, 91, 91]

  # recup points en bas à gauche -> 1
  # laisser en bas à droite à 0 _2

  # récup en haut à gauche -3
  # éventuellement en haut au milieu si existe -4

  # recup en haut à droite -6
  # laisser en haut opposéL à 0 -5

  # 8 est en bas à droite
  # laisser 7 à 0
  max_width = 0

  for i in range(len(cnts_poly)):
    if abs(cnts_poly[i][1] - MinY < 15):
      minx_local = cnts_poly[i][0]
      maxx_local = cnts_poly[i][0]
      for j in range(len(cnts_poly)):
        if ((abs(cnts_poly[j][1] - MinY) < 15) and (cnts_poly[j][0] <= minx_local)):
          minx_local = cnts_poly[j][0]
          indexes[5] = j
        if ((abs(cnts_poly[j][1] - MinY) < 15) and (cnts_poly[j][0] >= maxx_local)):
          maxx_local = cnts_poly[j][0]
          indexes[4] = j
    if abs(cnts_poly[i][1] - MaxY < 15):
      minx_local = cnts_poly[i][0]
      maxx_local = cnts_poly[i][0]
      for j in range(len(cnts_poly)):
        if ((abs(cnts_poly[j][1] - MaxY) < 20) and (cnts_poly[j][0] <= minx_local)):
          minx_local = cnts_poly[j][0]
          indexes[1] = j
        if ((abs(cnts_poly[j][1] - MaxY) < 20) and (cnts_poly[j][0] >= maxx_local)):
          maxx_local = cnts_poly[j][0]
          indexes[0] = j

    for j in range(len(cnts_poly)):
      if abs(cnts_poly[i][0] - cnts_poly[j][0]) > max_width and abs(cnts_poly[i][1] - cnts_poly[j][1] < 15):
        max_width = abs(cnts_poly[i][0] - cnts_poly[j][0])
        if (cnts_poly[i][0] < cnts_poly[j][0]):
          indexes[2] = j
          indexes[3] = i
        else:
          indexes[3] = j
          indexes[2] = i


  g_points = []

  for i, index in enumerate(indexes):
    if (indexes[i] != 91):
      g_points.append(cnts_poly[index])
    else:
      g_points.append(np.array([0, 0]))

  return g_points


def add_points(g_points_pos1, g_points_pos2, H12):

  pt1_f = np.array([int(g_points_pos2[1][0]), int(g_points_pos2[1][1]), 1]).reshape(3, 1)
  pt2_f = np.array([int(g_points_pos2[3][0]), int(g_points_pos2[3][1]), 1]).reshape(3,1)

  pt1_e = np.matmul(H12, pt1_f)
  pt2_e = np.matmul(H12, pt2_f)
  pt1_e /= pt1_e[2]
  pt2_e /=pt2_e[2]

  g_points_pos1[1]=np.array([int(pt1_e[0]), int(pt1_e[1])])
  g_points_pos1[3]=np.array([int(pt2_e[0]),int(pt2_e[1])])
  return g_points_pos1

def dist(point1, point2):
  return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def predict_dimensions_p1(g_points_pos2, values,  Hv):
  pt = []
  for i in values:
    ptf = np.array([int(g_points_pos2[i][0]), int(g_points_pos2[i][1]), 1]).reshape(3, 1)
    pte = np.matmul(Hv, ptf)
    pte/=pte[2]

    pt.append(np.array([int(pte[0]), int(pte[1])]))

  h = (dist(pt[0], pt[2])+dist(pt[1], pt[3]))/8 + 4.5
  l = (dist(pt[0], pt[1]) + dist(pt[2], pt[3]))/8

  return h, l

def draw_lines_pos1(image, g_points_pos1):
  #hauteur
  cv2.line(image, tuple(g_points_pos1[2]), tuple(g_points_pos1[0]), (0,255,0), 6)
  cv2.line(image, tuple(g_points_pos1[3]), tuple(g_points_pos1[1]), (0,255,0), 6)
  cv2.line(image, tuple(g_points_pos1[5]), tuple(g_points_pos1[7]), (0,255,0), 6)
  cv2.line(image, tuple(g_points_pos1[1]), tuple(g_points_pos1[0]), (0,255,0), 6)
  cv2.line(image, tuple(g_points_pos1[2]), tuple(g_points_pos1[3]), (0,255,0), 6)
  cv2.line(image, tuple(g_points_pos1[5]), tuple(g_points_pos1[3]), (0,255,0), 6)
  cv2.line(image, tuple(g_points_pos1[1]), tuple(g_points_pos1[7]), (0,255,0), 6)
  return image


def mask_processor(image1, mask1, image2, mask2, H12, Hv1_1, Hv1_2):
  mask1 = cv2.cvtColor(mask1, cv2.COLOR_RGB2GRAY)
  mask1 = cv2.resize(mask1, (640, 360))

  mask2 = cv2.cvtColor(mask2, cv2.COLOR_RGB2GRAY)
  mask2 = cv2.resize(mask2, (640, 480))
  mask2 = mask2[120:, :]

  image1 = cv2.resize(image1, (640, 360))

  cnts_poly_1 = segment_points(mask1)
  cnts_poly_1.reshape(len(cnts_poly_1), 2)
  good_points_pos1 = select_good_points_pos1(cnts_poly_1)

  cnts_poly_2 = segment_points(mask2)
  cnts_poly_2.reshape(len(cnts_poly_2), 2)
  good_points_pos2 = select_good_points_pos2(cnts_poly_2)
  print(good_points_pos2)

  good_points_pos1 = add_points(good_points_pos1, good_points_pos2, H12)
  image1 = draw_lines_pos1(image1, good_points_pos1)

  for i in range(len(good_points_pos1)):
    cv2.circle(image1, tuple(good_points_pos1[i]), 3, (255, 0, 0), 2)
  plt.imshow(image1, cmap = 'gray')
  plt.show()

  h1, l1 = predict_dimensions_p1(good_points_pos1, [0, 1, 2, 3], Hv1_1)
  h2, l2 = predict_dimensions_p1(good_points_pos2, [0, 1, 2, 3], Hv1_2)

  h = (h1 + h2) / 2
  l = (l1 + l2) / 2
  return image1, h, l