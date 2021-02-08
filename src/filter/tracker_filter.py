from settings import UNDETECTED_THRESH


def filter_undetected_trackers(trackers, attributes, detected_rects):

    for fid in trackers.keys():
        tracked_position = trackers[fid].get_position()
        t_left = int(tracked_position.left())
        t_top = int(tracked_position.top())
        t_right = int(tracked_position.right())
        t_bottom = int(tracked_position.bottom())

        # calculate the center point
        t_x_bar = 0.5 * (t_left + t_right)
        t_y_bar = 0.5 * (t_top + t_bottom)

        del_ret = True
        for (d_left, d_top, d_right, d_bottom) in detected_rects:
            if d_left <= t_x_bar <= d_right and d_top <= t_y_bar <= d_bottom:
                del_ret = False
                break
        if del_ret:
            attributes[fid]["undetected"] += 1

    del_ids = []
    for fid in trackers.keys():

        if attributes[fid]["undetected"] > UNDETECTED_THRESH:
            del_ids.append(fid)

    for idx in del_ids:
        attributes.pop(idx)
        trackers.pop(idx)

    return trackers, attributes


if __name__ == '__main__':

    filter_undetected_trackers(trackers={}, attributes={}, detected_rects=[])
