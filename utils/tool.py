import cv2
from settings import COCO_BODY_CONNECTIONS


def scale_up(pt, w, h):
    return int(pt[0] * w), int(pt[1] * h)


def transform_pt(pt, rx=1, ry=1, dx=0, dy=0):
    return pt[0] * rx + dx, pt[1] * ry + dy


def draw_key_points(img, key_points, skeletons=COCO_BODY_CONNECTIONS, show_key_points=True, show_skeletons=True,
                    show_key_number=True):
    show = img.copy()

    h, w = show.shape[:2]
    kps = key_points.copy()

    if show_skeletons:
        for p1i, p2i, color in skeletons:
            p1 = scale_up(pt=kps[p1i], w=w, h=h)
            p2 = scale_up(pt=kps[p2i], w=w, h=h)
            if p1 == (0, 0) or p2 == (0, 0):
                continue
            cv2.line(img=show, pt1=p1, pt2=p2, color=(255, 255, 255), thickness=2)

    if show_key_points:
        target_key_points = set([s[0] for s in skeletons] + [s[1] for s in skeletons])
        for i, kp in enumerate(kps):
            if i not in target_key_points:
                continue
            p = scale_up(pt=kp, w=w, h=h)
            if p == (0, 0):  # Joint wasn't detected
                continue
            cv2.circle(img=show, center=p, radius=3, color=(255, 255, 255), thickness=-1)

            if show_key_number:
                cv2.putText(img=show, text=str(i), org=p, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                            color=(0, 0, 255), thickness=2)
    return show


if __name__ == '__main__':
    draw_key_points(key_points=[], img=[])
