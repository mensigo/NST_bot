import os
import pathlib
import shutil


WEIGHTS_PATH = '../pytorch-CycleGAN-and-pix2pix/checkpoints'
RESULTS_PATH = '../pytorch-CycleGAN-and-pix2pix/results'

CONTENT_FOLDER_PATH = '../pytorch-CycleGAN-and-pix2pix/datasets/ukiyoe2photo/testB'
OUTPUT_FOLDER_PATH = '../pytorch-CycleGAN-and-pix2pix/results/style_ukiyoe_pretrained/test_latest/images'
INFERENCE_FILE_PATH = '../pytorch-CycleGAN-and-pix2pix/test.py'
NAME = 'style_ukiyoe_pretrained'


async def apply_GAN(content_path, save_path):

	# move content image to the special dir
    pathlib.Path(CONTENT_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
    shutil.move(content_path, os.path.join(CONTENT_FOLDER_PATH, 'content.jpg'))

    # download weights due to memory limits for /run
    os.makedirs(WEIGHTS_PATH + 'style_ukiyoe_pretrained', exist_ok=True)
    weights_fname = WEIGHTS_PATH + '/style_ukiyoe_pretrained/latest_net_G.pth'
    weights_url = 'http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/style_ukiyoe.pth'
    os.system(f'wget -N {weights_url} -O {weights_fname}')

    # inference
    command = f'python {INFERENCE_FILE_PATH}'
    command += f' --dataroot {CONTENT_FOLDER_PATH}'
    command += f' --results_dir {RESULTS_PATH}'
    command += f' --checkpoints_dir {WEIGHTS_PATH}'
    command += f' --name {NAME}'
    command += f' --model test'
    command += f' --no_dropout'
    os.system(command)

    # move result image back home
    shutil.move(os.path.join(OUTPUT_FOLDER_PATH, 'content_fake.png'), save_path)

    # delete content & result data, weights
    os.remove(os.path.join(CONTENT_FOLDER_PATH, 'content.jpg'))
    shutil.rmtree(RESULTS_PATH)
    os.remove(weights_fname)
