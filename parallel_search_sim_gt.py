# (c) 2021 Tian-Rui Chung
# this is single-use code to preprocess the data for the simulation
# don't waste time to reusing it
import logging
from multiprocessing import Pool
import os

import numpy as np
import tqdm
import gt_gen_sim
from sim import CalSimSimple, CalSimTrans3D
import cv2

THREADS = 12
DATASET_DIR = './datasets/sim'


def job(args):
    try:
        file, logger = args
        cal_sim = CalSimSimple(file=f'{file}/trace.txt')
        sims = cal_sim.split_strokes()
        #  trans_x, trans_y, angle, scale_x, scale_y, z_bias
        results = np.zeros((len(sims), 6), dtype=np.float32)
        masks = np.zeros((len(sims), 6), dtype=np.bool)
        fail_cnt = 0
        for i, sim in enumerate(sims):
            sim: CalSimTrans3D = CalSimTrans3D.from_cal_sim_simple(sim)
            target_image = cv2.imread(f'{file}/stroke_{i}.jpg', cv2.IMREAD_GRAYSCALE)
            params, loss = gt_gen_sim.search_params(sim, target_image)
            results[i, :] = params
            if loss > .5:
                logger.warning(f'loss too high: {loss} {file}/stroke_{i}.jpg')
                fail_cnt += 1
            else:
                masks[i, :] = True

        np.save(f'{file}/transform.npy', results)
        np.save(f'{file}/mask.npy', masks)
    except Exception as e:
        print(e)
        return 0, 0
    return fail_cnt, len(sims)


def main():
    logging.basicConfig(filename=f'sim_gt_gen.log', level=logging.WARNING)
    logger = logging.getLogger('sim_gt_gen')

    dirs = os.listdir(DATASET_DIR)
    with Pool(THREADS) as p:
        failed_cnt_list, total_list = zip(*list(tqdm.tqdm(
            p.imap(job,
                   [(os.path.join(DATASET_DIR, dir), logger)
                    for dir in dirs
                    if os.path.isdir(os.path.join(DATASET_DIR, dir))]), total=len(dirs))))
    failed_cnt = sum(failed_cnt_list)
    total = sum(total_list)
    full_chars = sum(1 if failed_cnt == 0 else 0 for failed_cnt in failed_cnt_list)
    print(f"{failed_cnt}/{total} ({failed_cnt/total:.6f}) strokes can't find all transform of strokes.")
    print(f"{full_chars}/{len(dirs)} ({full_chars/len(dirs):.6f}) chars find all transform.")


if __name__ == '__main__':
    main()
