import os
import pathlib
import shutil


# WEIGHTS_PATH = 'E:/Jupyter/#stepic/MIPT/final_project/pytorch-CycleGAN-and-pix2pix/checkpoints'
# CONTENT_FOLDER_PATH = 'E:/Jupyter/#stepic/MIPT/final_project/pytorch-CycleGAN-and-pix2pix/datasets/ukiyoe2photo/testB'
# SAVE_RESULT_PATH = 'E:/Jupyter/#stepic/MIPT/final_project/pytorch-CycleGAN-and-pix2pix/results'
# OUTPUT_FOLDER_PATH = 'E:/Jupyter/#stepic/MIPT/final_project/pytorch-CycleGAN-and-pix2pix/results/style_ukiyoe_pretrained/test_latest/images'
# INFERENCE_FILE_PATH = 'E:/Jupyter/#stepic/MIPT/final_project/pytorch-CycleGAN-and-pix2pix/test.py'
# NAME = 'style_ukiyoe_pretrained'

WEIGHTS_PATH = '../pytorch-CycleGAN-and-pix2pix/checkpoints'
RESULT_PATH = '../pytorch-CycleGAN-and-pix2pix/results'

CONTENT_FOLDER_PATH = '../pytorch-CycleGAN-and-pix2pix/datasets/ukiyoe2photo/testB'
OUTPUT_FOLDER_PATH = '../pytorch-CycleGAN-and-pix2pix/results/style_ukiyoe_pretrained/test_latest/images'
INFERENCE_FILE_PATH = '../pytorch-CycleGAN-and-pix2pix/test.py'
NAME = 'style_ukiyoe_pretrained'


async def apply_GAN(content_path, save_path):

	# move content image to the special dir
    pathlib.Path(CONTENT_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
    shutil.move(content_path, os.path.join(CONTENT_FOLDER_PATH, 'content.jpg'))

    # weights are predownloaded by a script

    # inference
    command = f'python {INFERENCE_FILE_PATH}'
    command += f' --dataroot {CONTENT_FOLDER_PATH}'
    command += f' --results_dir {RESULT_PATH}'
    command += f' --checkpoints_dir {WEIGHTS_PATH}'
    command += f' --name {NAME}'
    command += f' --model test'
    command += f' --no_dropout'
    os.system(command)

    # move result image back home
    shutil.move(os.path.join(OUTPUT_FOLDER_PATH, 'content_fake.png'), save_path)

    # delete content & result data
    os.remove(os.path.join(CONTENT_FOLDER_PATH, 'content.jpg'))
    shutil.rmtree(RESULT_PATH)