import sys
import os
import subprocess
import time
from pytube import YouTube
import ffmpeg
from pytube.exceptions import VideoUnavailable

class VidInfo:
    def __init__(self, yt_id, start_time, end_time, outdir):
        self.yt_id = yt_id
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.out_filename = os.path.join(outdir, yt_id + '_' + start_time + '_' + end_time + '.mp4')

def download(vidinfo):

    yt_base_url = 'https://www.youtube.com/watch?v='

    yt_url = yt_base_url+vidinfo.yt_id

    ydl_opts = {
        'format': '22/18',
        'quiet': True,
        'ignoreerrors': True,
        'no_warnings': True,
    }
    download_url = None
    try:
        print("Downloading: ", yt_url)
        time.sleep(2)
        youtube = YouTube(yt_url) 
        print("Youtube object created: ", youtube)

        video_streams = youtube.streams.filter(progressive=True, res="720p")
        print(video_streams)
        if video_streams:
            download_url = video_streams.first().url
        print(download_url)
    except VideoUnavailable as ex:
        print("VideoUnavailable: ", ex)
        return_msg = '{}, ERROR (youtube)!'.format(vidinfo.yt_id)
        return return_msg
   

    if download_url:
        if not os.path.exists(vidinfo.out_filename):
            subprocess.run([
                'ffmpeg', '-y', '-i', download_url, '-ss', str(vidinfo.start_time), '-to', str(vidinfo.end_time),
                '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k',
                # output to file
                vidinfo.out_filename
            ])
        else:
            print('Already downloaded: ', vidinfo.out_filename)

    return '{}, DONE!'.format(vidinfo.yt_id)


if __name__ == '__main__':
    data_dir = "../../data/input/avspeech/"
    split = sys.argv[1]
    csv_file = 'avspeech_{}.csv'.format(split)
    out_dir = os.path.join(data_dir, split)

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(data_dir, csv_file), 'r') as f:
        lines = f.readlines()
        lines = [x.split(',') for x in lines]
        vidinfos = [VidInfo(x[0], x[1], x[2], out_dir) for x in lines]

    for vidinfo in vidinfos:
        download(vidinfo)
