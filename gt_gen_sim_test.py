from gt_gen_sim import *
if __name__ == '__main__':
    cal_sim = CalSimSimple(file='test_strokes/char00482_stroke.txt')
    sims = cal_sim.split_strokes()
    # just test the first stroke
    sim: CalSimTrans3D = CalSimTrans3D.from_cal_sim_simple(sims[0])
    from_img = sim.get_image()
    cv2.imwrite('test_strokes/from_img_0.png', from_img)
    target_image = cv2.imread('test_strokes/tmp1_0.jpg', cv2.IMREAD_GRAYSCALE)

    params = search_params(sim, target_image)

    result = sim.transform(params)
    result_img = result.get_image()
    cv2.imwrite('test_strokes/result_img_0.png', result_img)

    print(params)
