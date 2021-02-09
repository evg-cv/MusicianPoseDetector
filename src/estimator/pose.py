import os
import time
import cv2
import math
import matplotlib.pyplot as plt

from ast import literal_eval
from src.detector.pose import PoseKeyDetection
from utils.tool import draw_key_points
from settings import TRACK_QUALITY, PERSON_TRACK_CYCLE, LOCAL, CUR_DIR


class PoseAnalyzer:

    def __init__(self, model_name='resnet101'):
        self.model_name = model_name
        self.class_pose_key = PoseKeyDetection(model_name=model_name, det_sz=None)
        self.graph = {}
        self.person_trackers = {}
        self.current_person_id = 1
        self.person_attributes = {}

    def detect_key_points(self, file_video):
        cap = cv2.VideoCapture(file_video)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cnt = 0

        while cap.isOpened():
            try:
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
                    # img_key_point = cv2.rectangle(img_key_point, (crop[0], crop[1]), (crop[2], crop[3]), (0, 255,
                    # 0), 2)

                cv2.imshow("Keypoints", img_key_point)
                print('Frame: {}/{}, Process time: {}'.format(cnt, length, time.time() - t1))
                cnt += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                    break
            except Exception as e:
                print(cnt, e)
                continue
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

        return

    def analyze_pose_attributes(self):
        person_key_points = {}
        if LOCAL:
            import pandas as pd
            person_key_points["Musician1"] = pd.read_csv("/media/main/Data/Task/MusicianPoseDetector/src/estimator/"
                                                         "musician_1_key_points.csv")
            person_key_points["Musician2"] = pd.read_csv("/media/main/Data/Task/MusicianPoseDetector/src/estimator/"
                                                         "musician_2_key_points.csv")
            person_key_points["Musician3"] = pd.read_csv("/media/main/Data/Task/MusicianPoseDetector/src/estimator/"
                                                         "musician_3_key_points.csv")
            person_key_points["Musician4"] = pd.read_csv("/media/main/Data/Task/MusicianPoseDetector/src/estimator/"
                                                         "musician_4_key_points.csv")
        else:
            for fid in self.person_attributes.keys():
                person_key_points[f"Musician{fid}"] = self.person_attributes[fid]["key_points"]

        pose_analysis = {}
        for p_key in person_key_points.keys():
            pose_analysis[p_key] = []
            if LOCAL:
                init_key_points = person_key_points[p_key].loc[0, :].values.tolist()
                rest_key_points = person_key_points[p_key].loc[1:, :].values.tolist()
            else:
                init_key_points = person_key_points[p_key][0]
                rest_key_points = person_key_points[p_key][1:]
            for key_points in rest_key_points:
                frame_diff = 0
                for i_key_point, key_point in zip(init_key_points, key_points):
                    if LOCAL:
                        i_x, i_y, _ = literal_eval(i_key_point)
                        x, y, _ = literal_eval(key_point)
                    else:
                        i_x, i_y, _ = i_key_point
                        x, y, _ = key_point
                    frame_diff += math.sqrt((i_x - x) ** 2 + (i_y - y) ** 2)
                frame_diff /= 17
                pose_analysis[p_key].append(frame_diff)

        output_graph = os.path.join(CUR_DIR, "result.jpg")
        legends = list(pose_analysis.keys())
        figure, ax = plt.subplots()
        for a_key in pose_analysis.keys():
            plt.plot(pose_analysis[a_key], linewidth=3.0)
        plt.legend(legends, fontsize=8)
        plt.xlabel('Frames', fontsize=16)
        plt.ylabel('Average Motion', fontsize=16)
        plt.title('Average Motion - Frame per Musician', fontsize=16)
        plt.show()
        figure.savefig(output_graph)
        print(f"[INFO] Successfully saved the result graph in {output_graph}")

        return


if __name__ == '__main__':
    video_path1 = ''
    crop_mode = 'adaptive'
    class_main = PoseAnalyzer('resnet101')
    class_main.analyze_pose_attributes()
