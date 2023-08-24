import os
import cv2
from sim import CalSimSimple


def main():
    trace_file = "char00554_stroke.txt"
    output_dir = "char00554_stroke"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    full_sim = CalSimSimple(file=trace_file)
    full_img = full_sim.get_image()
    cv2.imwrite(os.path.join(output_dir, "full.png"), full_img)
    strokes_sim = full_sim.split_strokes()
    for i, sim in enumerate(strokes_sim):
        im = sim.get_image()
        cv2.imwrite(f"{output_dir}/stroke_{i}.png", im)

if __name__ == '__main__':
    main()