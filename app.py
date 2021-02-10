import os
import glob

from src.estimator.pose import PoseAnalyzer
from settings import VIDEO_PATH, OUTPUT_DIR, FILE_NAME


if __name__ == '__main__':

    for c_file in glob.glob(os.path.join(OUTPUT_DIR, '*.*')):
        os.remove(c_file)
    person_key_points = PoseAnalyzer()
    person_key_points.detect_key_points(file_video=VIDEO_PATH, file_name=FILE_NAME)
    person_key_points.analyze_pose_attributes()
