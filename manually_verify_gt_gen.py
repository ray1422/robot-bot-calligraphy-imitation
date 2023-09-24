import cv2
import numpy as np
import sim

TEST_DIR = './datasets/sim/5a2d'
canvas = np.ones((256, 256), dtype=np.uint8) * 255
sims = sim.CalSimSimple(file=f'{TEST_DIR}/trace.txt').split_strokes()
for i, my_sim in enumerate(sims):
    try:
        trans = np.load(f'{TEST_DIR}/transform_{i}.npy')
        my_sim = sim.CalSimTrans3D.from_cal_sim_simple(my_sim).transform(trans[i, :])
        img = my_sim.get_image()
        # merge black pixels
        canvas[img < 100] = img[img < 100]
    except FileNotFoundError:
        break
    

cv2.imwrite('outputs/trans_gt_gen_test.png', canvas)