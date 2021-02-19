import os
import glob
import ntpath
import pandas as pd


avg_motion_csv_files_dir = "/media/main/Data/Task/MusicianPoseDetector/43/avg"
reference_file_path = "/media/main/Data/Task/MusicianPoseDetector/43/43.csv"
output_csv_file_path = "convert.csv"


def convert_average_motion():
    csv_file_paths = glob.glob(os.path.join(avg_motion_csv_files_dir, "*.csv"))
    ref_content = pd.read_csv(reference_file_path)
    beat_contents = ref_content["beat"].values.tolist()
    cnt = 0
    result = {"time": [], "beat": []}
    for j, c_file in enumerate(csv_file_paths):
        person_id = ntpath.basename(c_file).replace(".csv", "").split("_")[1]
        result[f"person_{person_id}"] = []
        average_motions = pd.read_csv(c_file).values.tolist()
        for i, avg_motion in enumerate(average_motions):
            if i % 3 == 0:
                _, avg_motion_val = avg_motion
                result[f"person_{person_id}"].append(avg_motion_val)
                if j == 0:
                    result["time"].append(0.1 * cnt)
                    if cnt >= len(beat_contents):
                        result["beat"].append(0)
                    else:
                        result["beat"].append(beat_contents[cnt])
                    cnt += 1
        lens = []
        for r_key in result.keys():
            lens.append(len(result[r_key]))
        correction_len = min(lens)
        for r_key in result.keys():
            result[r_key] = result[r_key][:correction_len]

    pd.DataFrame(result, columns=list(result.keys())).to_csv(output_csv_file_path, header=True, index=True, mode='w')
    print(f"[INFO] Successfully saved in {output_csv_file_path}")

    return


if __name__ == '__main__':
    convert_average_motion()
