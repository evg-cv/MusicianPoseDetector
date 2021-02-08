import time
import cv2

from src.detector.pose import PoseKeyDetection
from utils.tool import draw_key_points
from settings import TRACK_QUALITY, PERSON_TRACK_CYCLE, LOCAL


class PoseAnalyzer:

    def __init__(self, model_name='resnet101'):
        self.model_name = model_name
        self.class_pose_key = PoseKeyDetection(model_name=model_name, det_sz=None)
        # self.class_anal = Analyzer()
        self.graph = {}
        self.person_trackers = {}
        self.current_person_id = 1
        self.person_attributes = {}
        # self.font = ImageFont.truetype("font/arial.ttf", 20)

        # self.img_logo = cv2.resize(cv2.imread('logo/logo.png'), None, fx=LOGO_RESIZE, fy=LOGO_RESIZE)

    def detect_key_points(self, file_video):
        cap = cv2.VideoCapture(file_video)
        frame_pos_list = []
        key_points_list = []

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cnt = 0

        while cap.isOpened():
            t1 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            fids_to_delete = []
            for fid in self.person_trackers.keys():
                tracking_quality = self.person_trackers[fid].update(frame)

                # If the tracking quality is good enough, we must delete this tracker
                if tracking_quality < TRACK_QUALITY:
                    fids_to_delete.append(fid)

            for fid in fids_to_delete:
                print("Removing fid " + str(fid) + " from list of trackers")
                self.person_trackers.pop(fid, None)
                self.person_attributes.pop(fid, None)
            if cnt % PERSON_TRACK_CYCLE == 0:
                self.person_trackers, self.person_attributes, self.current_person_id, result_img = \
                    self.class_pose_key.process_frame(frame=frame, trackers=self.person_trackers,
                                                      attributes=self.person_attributes,
                                                      person_id=self.current_person_id)
            else:
                result_img, self.person_attributes = \
                    self.class_pose_key.track_persons(person_frame=frame, trackers=self.person_trackers,
                                                      attributes=self.person_attributes)
            img_key_point = result_img
            # ------------------------ draw result ---------------------
            for fid in self.person_attributes.keys():
                key_points = self.person_attributes[fid]["key_points"][-1]
                img_key_point = draw_key_points(img_key_point, key_points=key_points)
                # img_key_point = cv2.rectangle(img_key_point, (crop[0], crop[1]), (crop[2], crop[3]), (0, 255, 0), 2)

            cv2.imshow("Keypoints", img_key_point)
            print('Frame: {}/{}, Process time: {}'.format(cnt, length, time.time() - t1))
            cnt += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break
        # kill open cv things
        cap.release()
        cv2.destroyAllWindows()
        if LOCAL:
            import pandas as pd
            for fid in self.person_attributes.keys():
                key_points = self.person_attributes[fid]["key_points"]
                headers = [f"KeyPoint{i}" for i in range(17)]
                pd.DataFrame(key_points, columns=headers).to_csv(f"musician_{fid}_key_points.csv",
                                                                 header=True, index=False, mode="w")

        return frame_pos_list, key_points_list


if __name__ == '__main__':
    # ---------------------- source --------------------------------------------------------------------------
    video_path1 = '/media/main/Data/Task/MusicianPoseDetector/yt1s.com - URMP  32  The Art of the Fugue_v720P.mp4'

    # crop_mode = [100, 0, 620, 700]
    crop_mode = 'adaptive'
    # crop_mode = 'full'
    class_main = PoseAnalyzer('resnet101')
    class_main.detect_key_points(file_video=video_path1)
