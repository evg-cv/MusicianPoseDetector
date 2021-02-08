import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PERSON_MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'frozen_inference_graph.pb')

# -------------------------- KEYPOINTS --------------------------------------------------------------
"""Body part locations in the 'coordinates' list."""
Nose = 0
LEye = 1
REye = 2
LEar = 3
REar = 4
LShoulder = 5
RShoulder = 6
LElbow = 7
RElbow = 8
LWrist = 9
RWrist = 10
LHip = 11
RHip = 12
LKnee = 13
RKnee = 14
LAnkle = 15
RAnkle = 16
CocoPart = [Nose, LEye, REye, LEar, REar, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee,
            RKnee, LAnkle, RAnkle]

# -------------------------- SKELETONS ------------------------------------------------------------
# Body part locations in the 'coordinates' list.
HEAD_CONNECTIONS = [(Nose, LEye, (210, 182, 247)), (Nose, REye, (127, 127, 127)), (LEye, REye, (194, 119, 227)),
                    (LEye, LEar, (199, 199, 199)), (LEar, LShoulder, (141, 219, 219)),
                    (REye, REar, (34, 189, 188)), (REar, RShoulder, (207, 190, 23))]
UPPER_BODY_CONNECTIONS = [(LShoulder, RShoulder, (150, 152, 255)), (LShoulder, LHip, (138, 223, 152)),
                          (RShoulder, RHip, (40, 39, 214)), (LHip, RHip, (44, 160, 44))]
RIGHT_ARM_CONNECTIONS = [(RShoulder, RElbow, (213, 176, 197)), (RElbow, RWrist, (148, 156, 196))]
RIGHT_LEG_CONNECTIONS = [(RHip, RKnee, (120, 187, 255)), (RKnee, RAnkle, (14, 127, 255))]
LEFT_ARM_CONNECTIONS = [(LShoulder, LElbow, (189, 103, 148)), (LElbow, LWrist, (75, 86, 140))]
LEFT_LEG_CONNECTIONS = [(LHip, LKnee, (232, 199, 174)), (LKnee, LAnkle, (180, 119, 31))]
RIGHT_ARM_DIRECT_CONNECTIONS = [(RShoulder, RWrist, (50, 220, 50))]
LEFT_ARM_DIRECT_CONNECTIONS = [(LShoulder, LWrist, (220, 50, 50))]
COCO_BODY_CONNECTIONS = HEAD_CONNECTIONS + UPPER_BODY_CONNECTIONS + \
                        RIGHT_ARM_CONNECTIONS + LEFT_ARM_CONNECTIONS + \
                        RIGHT_LEG_CONNECTIONS + LEFT_LEG_CONNECTIONS

PERSON_THRESHOLD = 0.3
DISPLAY_SCALE = 0.5
DISPLAY_COMPARE_SCALE = 0.3
POSE_MARGIN_X = 0.1
POSE_MARGIN_Y = 0.02
OVERLAP_THRESH = 0.7
UNDETECTED_THRESH = 3
MARGIN = 0
CIRCLE_RADIUS = 8
TRACK_QUALITY = 2
PERSON_TRACK_CYCLE = 20

LOCAL = True
