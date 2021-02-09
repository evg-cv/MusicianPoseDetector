from utils.tool import transform_pt
from settings import CocoPart


class CocoPose:
    key_points = [[0.0, 0.0, 0.0] for _ in CocoPart]

    def __init__(self, key_points):
        assert len(key_points) == len(CocoPart)
        for part in CocoPart:

            if key_points[part] is None:
                continue

            key_point = key_points[part]
            if len(key_point) >= 3:
                self.key_points[part] = [key_point[0], key_point[1], key_point[2]]
            elif len(key_point) < 3:
                self.key_points[part][:2] = [key_point[0], key_point[1]]

    def transform(self, rx=1, ry=1, dx=0, dy=0):
        key_points = []
        for part in CocoPart:
            pt = self.key_points[part]
            score = self.key_points[part][2]
            new_pt = transform_pt(pt=pt, rx=rx, ry=ry, dx=dx, dy=dy)
            key_points.append([new_pt[0], new_pt[1], score])

        self.key_points = key_points
        return self.key_points


if __name__ == '__main__':
    CocoPose(key_points={}).transform()
