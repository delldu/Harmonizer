"""Image/Video Autops Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import torch

import todos
from . import harmonizer

import pdb


def get_harmonize_model():
    """Create model."""
    model = harmonizer.Harmonizer()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;
    # torch::jit::setTensorExprFuserEnabled(false);
    todos.data.mkdir("output")
    if not os.path.exists("output/image_harmonize.torch"):
        model.save("output/image_harmonize.torch")

    print(f"Running model on {device} ...")

    return model, device


def image_harmonize_predict(input_files, mask_files, output_dir):
    from tqdm import tqdm

    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_harmonize_model()

    # load files
    image_filenames = todos.data.load_files(input_files)
    mask_filenames = todos.data.load_files(mask_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename, maskname in zip(image_filenames, mask_filenames):
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        B, C, H, W = input_tensor.size()
        mask_tensor = todos.data.load_tensor(maskname)
        mask_tensor = mask_tensor[:, 0:1, :, :]

        orig_tensor = input_tensor.clone().detach()
        with torch.no_grad():
            predict_tensor = model(input_tensor.to(device), mask_tensor.to(device))
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()
