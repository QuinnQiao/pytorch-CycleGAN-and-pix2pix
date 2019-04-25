import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torchvision.utils as vutils
import torch

def save_images(save_dir, save_name, images, num_input):
    image_tensor = torch.cat(images, 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=num_input, padding=0, normalize=True)
    vutils.save_image(image_grid, os.path.join(save_dir, save_name), nrow=1)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.phase = 'valid'   # for create_dataset, use dataroot/valid*
    opt.model = 'cycle_gan'
    opt.dataset_mode = 'unaligned'

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # use neither of dropout or BN
    model.eval()
    
    images_in_A, images_in_B = [], []
    images_out_A, images_out_B = [], []
    for data in dataset:
        images_in_A.append((data['A']+1)/2)
        images_in_B.append((data['B']+1)/2)
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        images_out_A.append((visuals['fakeB'].cpu().data+1)/2)
        images_out_B.append((visuals['fakeA'].cpu().data+1)/2)

    num_input = len(images_in_A)

    a2b = images_in_A + images_out_A
    b2a = images_in_B + images_out_B
    
    save_images(opt.result_dir, 'A2B.jpg', a2b, num_input)
    save_images(opt.result_dir, 'B2A.jpg', b2a, num_input)