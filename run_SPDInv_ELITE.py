import argparse
from ELITE.elite_utils import ELITE_inversion_and_edit

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="images/000000000002.jpg")  # the editing category that needed to run
    parser.add_argument('--output', type=str, default="outputs")  # the editing category that needed to run
    parser.add_argument('--source', type=str, default="a photo of a cat")
    parser.add_argument('--target', type=str, default="a photo of a white *")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/global_mapper.pt")
    args = parser.parse_args()

    param = {}
    param['image_path'] = args.input
    param['output_dir'] = args.output
    param['ref_image_path'] = args.input
    param['prompt_src'] = args.source
    param['prompt_tar'] = args.target
    param['checkpoint_path'] = args.checkpoint

    ELITE_inversion_and_edit(**param)