from src.estimator.pose import PoseAnalyzer
from settings import VIDEO_PATH


if __name__ == '__main__':
    person_key_points = PoseAnalyzer()
    person_key_points.detect_key_points(file_video=VIDEO_PATH)
    person_key_points.analyze_pose_attributes()
