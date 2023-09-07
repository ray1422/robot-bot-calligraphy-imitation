from gt_gen_sim import *
if __name__ == '__main__':
    cal_sim = CalSimSimple(file='test_strokes/char00600_stroke.txt')
    sims = cal_sim.split_strokes()
    for i in range(9):
        sim: CalSimTrans3D = CalSimTrans3D.from_cal_sim_simple(sims[i])
        from_img = sim.get_image()
        cv2.imwrite(f'test_strokes/from_img_{i}.png', from_img)
        target_image = cv2.imread(f'test_strokes/tmp1_{i}.jpg', cv2.IMREAD_GRAYSCALE)

        params, loss = search_params(sim, target_image)
        result = sim.transform(params)
        result_img = result.get_image()
        cv2.imwrite(f'test_strokes/result_img_{i}.png', result_img)

        print(params, loss)
