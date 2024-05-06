import sys
sys.path.append('../mash-autoencoder')

import torch

from mash_autoencoder.Module.Convertor.mash_split import Convertor

from py_kan.Dataset.mash_kan import MashKANDataset
from py_kan.Model.KAN import KAN
from py_kan.Module.trainer import Trainer


def demo():
    dataset_root_folder_path = "/home/chli/Dataset/"
    accum_iter = 1
    model_file_path = "./output/pretrain-4dim/model_last.pth"
    model_file_path = None
    dtype = torch.float32
    device = "cpu"
    warm_epoch_step_num = 100
    warm_epoch_num = 0
    finetune_step_num = 100000000
    lr = 1e-10
    weight_decay = 1e-10
    factor = 0.99
    patience = 10000
    min_lr = 1e-6
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

    train_scale = 0.9
    val_scale = 0.1

    convertor = Convertor(dataset_root_folder_path)
    convertor.convertToSplitFiles(train_scale, val_scale)

    if False:
        model = KAN(width=[22, 11, 22], grid=3, k=3)

        kan_dataset = MashKANDataset('/home/chli/Dataset/').toKANDataset(40)

        model.train(kan_dataset, steps=10000, lr=1e-5, batch=1)
        exit()

    trainer = Trainer(
        dataset_root_folder_path,
        accum_iter,
        model_file_path,
        dtype,
        device,
        warm_epoch_step_num,
        warm_epoch_num,
        finetune_step_num,
        lr,
        weight_decay,
        factor,
        patience,
        min_lr,
        save_result_folder_path,
        save_log_folder_path,
    )

    trainer.autoTrain()
    return True
