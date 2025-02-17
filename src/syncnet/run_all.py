import os
import argparse
from syncnet.run_pipeline import run_pipeline
from syncnet.run_syncnet import run_syncnet
from tempfile import TemporaryDirectory
from pathlib import Path


def main(video_path: str) -> float:
    syncnet_model_path = Path(os.environ["SYNCNET_MODEL_DIR"]) / "syncnet_v2.model"
    assert syncnet_model_path.exists()

    with TemporaryDirectory() as tmp_dir:
        pipeline_opts = argparse.Namespace(
            data_dir=tmp_dir,
            videofile=video_path,
            reference="",
            facedet_scale=0.25,
            crop_scale=0.4,
            min_track=100,
            frame_rate=25,
            num_failed_det=25,
            min_face_size=100,
        )
        run_pipeline(pipeline_opts)

        processed_video_path = Path(tmp_dir) / "pycrop" / "00000.avi"
        assert processed_video_path.exists()

        syncnet_opts = argparse.Namespace(
            initial_model=os.path.join(os.environ["SYNCNET_MODEL_DIR"], "syncnet_v2.model"),
            batch_size=20,
            vshift=1,
            data_dir=tmp_dir,
            videofile=str(processed_video_path),
            reference="",
        )
        return run_syncnet(syncnet_opts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "SyncNet")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    args = parser.parse_args()
    result = main(**vars(args))
