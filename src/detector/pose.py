import numpy as np
import cv2
import dlib
import collections

from src.filter.nms import non_max_suppression_slow
from src.filter.tracker_filter import filter_undetected_trackers
from src.detector.person import PersonDetector
from src.detector.openpifpaf_pose_detector import OpenPifPafPoseDetector
from settings import *


class PoseKeyDetection:

    def __init__(self, model_name='resnet50', det_sz=None):
        if det_sz is None:
            det_sz = (360, 360)

        self.key_point_detector = OpenPifPafPoseDetector(det_sz=det_sz, checkpoint=model_name)
        self.people_detector = PersonDetector()
        self.det_sz = det_sz

    def detect_key_points(self, frame, top, bottom, left, right, im_w, im_h):
        key_points = None
        crop = frame[top:bottom, left:right]
        dx, dy = left / im_w, top / im_h  # offset
        rx, ry = (right - left) / im_w, (bottom - top) / im_h  # scale

        # detect key_points
        crop = cv2.resize(crop, self.det_sz)
        pose = self.key_point_detector.detect(img=crop)

        # write to csv data
        if pose is not None:
            key_points = pose.transform(rx=rx, ry=ry, dx=dx, dy=dy)

        return key_points

    def track_persons(self, person_frame, trackers, attributes):

        height, width = person_frame.shape[:2]

        all_track_rects = []
        all_track_keys = []

        for fid in trackers.keys():
            tracked_position = trackers[fid].get_position()
            t_left = int(tracked_position.left())
            t_top = int(tracked_position.top())
            t_right = int(tracked_position.right())
            t_bottom = int(tracked_position.bottom())
            all_track_rects.append([t_left, t_top, t_right, t_bottom])
            all_track_keys.append(fid)

        filter_ids = non_max_suppression_slow(boxes=np.array(all_track_rects), keys=all_track_keys)

        for idx in filter_ids:
            attributes.pop(idx)
            trackers.pop(idx)

        for fid in trackers.keys():
            tracked_position = trackers[fid].get_position()
            t_left = int(tracked_position.left())
            t_top = int(tracked_position.top())
            t_right = int(tracked_position.right())
            t_bottom = int(tracked_position.bottom())

            key_points = self.detect_key_points(frame=person_frame, top=t_top, bottom=t_bottom, left=t_left,
                                                right=t_right, im_w=width, im_h=height)

            attributes[fid]["key_points"].append(key_points)

            # cv2.circle(person_frame, (t_center_x, t_center_y), CIRCLE_RADIUS, (0, 0, 255), -1)
            # cv2.rectangle(person_frame, (int(w_ratio * t_left), int(h_ratio * t_top)),
            #               (int(w_ratio * t_right), int(h_ratio * t_bottom)), (0, 0, 255), 3)
            #
            cv2.putText(person_frame, "Musician{}".format(str(attributes[fid]["id"])), (t_left, t_top),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        return person_frame, attributes

    def process_frame(self, frame, trackers, attributes, person_id):
        im_h, im_w = frame.shape[:2]
        rect_list, _, _ = self.people_detector.detect_from_images(frame)
        detected_centers = []

        for coordinates in rect_list:
            left, top, right, bottom = coordinates
            if right > im_w - 10 or left < 10:
                pass
            # left = max(int((left - POSE_MARGIN_X * im_w)), 0)
            # top = max(int((top - POSE_MARGIN_Y * im_h)), 0)
            # right = min(int((right + POSE_MARGIN_X * im_w)), im_w)
            # bottom = min(int((bottom + POSE_MARGIN_Y * im_h)), im_h)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            x_bar = 0.5 * (left + right)
            y_bar = 0.5 * (top + bottom)
            detected_centers.append([left, top, right, bottom])
            key_points = self.detect_key_points(frame=frame, left=left, right=right, top=top, bottom=bottom,
                                                im_w=im_w, im_h=im_h)

            matched_fid = None

            for fid in trackers.keys():

                tracked_position = trackers[fid].get_position()
                t_left = int(tracked_position.left())
                t_top = int(tracked_position.top())
                t_right = int(tracked_position.right())
                t_bottom = int(tracked_position.bottom())

                # calculate the center point
                t_x_bar = 0.5 * (t_left + t_right)
                t_y_bar = 0.5 * (t_top + t_bottom)

                # check if the center point of the face is within the rectangle of a tracker region.
                # Also, the center point of the tracker region must be within the region detected as a face.
                # If both of these conditions hold we have a match

                if t_left <= x_bar <= t_right and t_top <= y_bar <= t_bottom and left <= t_x_bar <= right \
                        and top <= t_y_bar <= bottom:
                    matched_fid = fid
                    trackers.pop(fid)
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(frame, dlib.rectangle(left - MARGIN, top - MARGIN, right + MARGIN,
                                                              bottom + MARGIN))
                    trackers[matched_fid] = tracker
                    attributes[matched_fid]["undetected"] = 0
                    attributes[matched_fid]["key_points"].append(key_points)

                    # cv2.circle(frame, (t_x_bar, t_y_bar), CIRCLE_RADIUS, (0, 0, 255), -1)
                    # cv2.rectangle(show_img, (int(w_ratio * t_left), int(h_ratio * t_top)),
                    #               (int(w_ratio * t_right), int(h_ratio * t_bottom)), (0, 0, 255), 3)
                    #
                    cv2.putText(frame, "Musician{}".format(str(attributes[matched_fid]["id"])),
                                (t_left, t_top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            # If no matched fid, then we have to create a new tracker
            if matched_fid is None:
                print("Creating new tracker " + str(person_id))
                # Create and store the tracker
                tracker = dlib.correlation_tracker()
                tracker.start_track(frame, dlib.rectangle(left - MARGIN, top - MARGIN, right + MARGIN,
                                                          bottom + MARGIN))
                trackers[person_id] = tracker

                temp_dict = collections.defaultdict()
                temp_dict["id"] = str(person_id)
                temp_dict["undetected"] = 0
                temp_dict["key_points"] = [key_points]
                attributes[person_id] = temp_dict
                # cv2.circle(frame, (x_bar, y_bar), CIRCLE_RADIUS, (0, 0, 255), -1)
                # cv2.rectangle(show_img, (int(w_ratio * left), int(h_ratio * top)),
                #               (int(w_ratio * right), int(h_ratio * bottom)), (0, 0, 255), 3)
                #
                cv2.putText(frame, "Musician{}".format(str(attributes[person_id]["id"])),
                            (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                #
                # Increase the currentFaceID counter
                person_id += 1

        trackers, attributes = filter_undetected_trackers(trackers=trackers, attributes=attributes,
                                                          detected_rects=detected_centers)

        return trackers, attributes, person_id, frame


if __name__ == '__main__':
    PoseKeyDetection().process_frame(frame=cv2.imread(""), attributes={}, trackers={}, person_id=0)
