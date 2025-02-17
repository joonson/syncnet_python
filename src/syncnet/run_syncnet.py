#!/usr/bin/python
#-*- coding: utf-8 -*-

import argparse, os, glob

from syncnet.SyncNetInstance import SyncNetInstance
import numpy as np


def run_syncnet(opt) -> float:
    setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
    setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
    setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
    setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))

    # ==================== LOAD MODEL AND FILE LIST ====================

    s = SyncNetInstance()

    s.loadParameters(opt.initial_model)
    print("Model %s loaded."%opt.initial_model)

    flist = glob.glob(os.path.join(opt.crop_dir,opt.reference,'0*.avi'))
    flist.sort()

    # ==================== GET OFFSETS ====================

    dists = []
    for idx, fname in enumerate(flist):
        *_, dist = s.evaluate(opt,videofile=fname)
        dists.append(dist)

    dists: np.ndarray = np.asarray(dists)[0]
    no_time_shift_idx = dists.shape[-1] // 2
    dists = dists[:, no_time_shift_idx]
    result = dists.mean()
    print(f"Syncnet mean distance: {result}")
    return result


if __name__ == "__main__":
    initial_model = os.path.join(os.environ["SYNCNET_MODEL_DIR"], "syncnet_v2.model")

    parser = argparse.ArgumentParser(description = "SyncNet")
    parser.add_argument('--initial_model', type=str, default=initial_model, help='')
    parser.add_argument('--batch_size', type=int, default='20', help='')
    parser.add_argument('--vshift', type=int, default='1', help='')
    parser.add_argument('--data_dir', type=str, default='data/work', help='')
    parser.add_argument('--videofile', type=str, default='', help='')
    parser.add_argument('--reference', type=str, default='', help='')
    opt = parser.parse_args()

    run_syncnet(opt)
