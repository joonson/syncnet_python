import os
import glob
import numpy as np


def main(video_folder_name, video_extension_name='mp4'):
    print(os.path.join(video_folder_name, f"*{video_extension_name}"))
    video_list = glob.glob(os.path.join(video_folder_name, f"*{video_extension_name}"))
    assert len(video_list) > 0, f"No video found in {video_folder_name} with extension {video_extension_name}, please double check the video folder name and extension name."

    # ==================== PREPARE RESULT FILE ====================
    filename = './result.txt'
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as f:
        f.write("video_filename av_offset confidence_score dist_min dist_max dist_mean dist_median\n")
    f.close()

    # ==================== RUN SYNCNET ====================
    for i in range(len(video_list)):
        os.system(f"python run_pipeline.py --videofile {video_list[i]} --reference {os.path.split(video_list[i])[-1][:-4]} --data_dir ./output")
        os.system(f"python run_syncnet.py --videofile {video_list[i]} --reference {os.path.split(video_list[i])[-1][:-4]} --data_dir ./output")
        # print(video_list[i])
    
    # ==================== CALCULATE RESULTS TO FILE ====================
    results = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            metrics_list = lines[i].split()
            metrics_list = np.array([float(i) for i in metrics_list[1:]], dtype=np.float32)
            results.append(metrics_list)
    f.close()
    results = np.array(results)
    results = np.mean(results, axis=0)
    with open(filename, 'a') as f:
        f.write(f"Mean Average Score {' '.join([str(i) for i in results])}\n")
    f.close()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "SyncNet")
    parser.add_argument('--video_folder_name', type=str, default='./data', help='')
    parser.add_argument('--video_extension_name', type=str, default='mp4', help='')
    opt = parser.parse_args()

    main(opt.video_folder_name, opt.video_extension_name)