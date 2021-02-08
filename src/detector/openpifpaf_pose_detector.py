import argparse
import cv2
import torch
import openpifpaf

from PIL import Image
from utils.coco_pose import CocoPose


class OpenPifPafPoseDetector(object):
    def __init__(self, det_sz, checkpoint='resnet50', device='cpu', force_complete_pose=True, instance_threshold=0.2,
                 seed_threshold=0.5):
        """

        :param det_sz:
        :param device:  cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu
        :param force_complete_pose:
        :param instance_threshold:
        :param seed_threshold:
        """
        parser = argparse.ArgumentParser()
        openpifpaf.decoder.cli(parser)
        openpifpaf.network.nets.cli(parser)

        args = parser.parse_args()
        args.force_complete_pose = force_complete_pose
        args.instance_threshold = instance_threshold
        args.seed_threshold = seed_threshold
        args.checkpoint = checkpoint

        self.width, self.height = det_sz
        # Load model
        self.model, _ = openpifpaf.network.nets.factory_from_args(args)
        self.model = self.model.to(device)
        self.processor = openpifpaf.decoder.factory_from_args(args, self.model)
        self.device = device

    def __detect_key_points(self, im):
        im_sz = im.shape[:2][::-1]
        target_wh = (self.width, self.height)
        if (im_sz[0] > im_sz[1]) != (target_wh[0] > target_wh[1]):
            target_wh = (target_wh[1], target_wh[0])
        if im_sz[0] != target_wh[0] or im_sz[1] != target_wh[1]:
            print('!!! have to resize image to {} from {}'.format(target_wh, im_sz))
            im = cv2.resize(im, target_wh, cv2.INTER_CUBIC)

        im_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        preprocess = openpifpaf.datasets.transforms.EVAL_TRANSFORM
        processed_image_cpu, _, __ = preprocess(im_pil, [], None)
        processed_image = processed_image_cpu.contiguous().to(self.device, non_blocking=True)

        all_fields = self.processor.fields(torch.unsqueeze(processed_image.float(), 0))[0]
        keypoint_sets, scores = self.processor.keypoint_sets(all_fields)

        # Normalize scale
        keypoint_sets[:, :, 0] /= processed_image_cpu.shape[2]
        keypoint_sets[:, :, 1] /= processed_image_cpu.shape[1]

        """
            Save keypoint coordinates of the *first* human pose identified to a CSV file.

            Coordinates are scaled to refer the resized image.

            Columns are in order frame_no, nose.(x|y|p), (l|r)eye.(x|y|p), (l|r)ear.(x|y|p), (l|r)shoulder.(x|y|p),
            (l|r)elbow.(x|y|p), (l|r)wrist.(x|y|p), (l|r)hip.(x|y|p), (l|r)knee.(x|y|p), (l|r)ankle.(x|y|p)

            l - Left side of the identified joint
            r - Right side of the identified joint
            x - X coordinate of the identified joint
            y - Y coordinate of the identified joint
            p - Probability of the identified joint
        """
        return keypoint_sets.tolist(), scores.tolist()

    @staticmethod
    def __parse_output(keypoint_sets):
        if len(keypoint_sets) == 0:
            return None
        else:
            # return first person and its pose
            keypoints = []
            for kpt in keypoint_sets[0]:
                keypoints.append([kpt[0], kpt[1], kpt[2]])

            return CocoPose(key_points=keypoints)

    def detect(self, img):
        keypoint_sets, pose_scores = self.__detect_key_points(im=img)

        return self.__parse_output(keypoint_sets)
