
from enum import Enum
import numpy as np

class HomographyType(Enum):
    UNKNOWUN = -1
    NORMAL = 0
    CONCAVE = 1
    TWIST = 2
    REFLECTION = 3

    def __str__(self) -> str:
        return str(self.name)

def classifyHomography(pts1: np.ndarray, pts2: np.ndarray) -> int:
    if len(pts1) != 4 or len(pts2) != 4: 
        return HomographyType.UNKNOWUN

    pt1 = np.cross(pts1 - np.roll(pts1, -1, axis=0), pts1 - np.roll(pts1, 1, axis=0))
    pt2 = np.cross(pts2 - np.roll(pts2, -1, axis=0), pts2 - np.roll(pts2, 1, axis=0))

    pts = pt1 * pt2
    h_type = (pts < 0).sum()
    if h_type == 4:
        return HomographyType.REFLECTION
    elif h_type == 2:
        return HomographyType.TWIST
    elif h_type in [1, 3]:
        return HomographyType.CONCAVE
    return HomographyType.NORMAL

def polyArea(points):
    if type(points) == np.ndarray:
        return polyArea_vector(points)
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
def polyArea_vector(points: np.ndarray):

    right_shift_points = np.roll(points, 1, axis=0)
    area = np.cross(points, right_shift_points)
    return abs(area.sum()) / 2.0
