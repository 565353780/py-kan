import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class MashKANDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        preload_data: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.preload_data = preload_data

        self.mash_folder_path = self.dataset_root_folder_path + "MashV3/"
        self.split_folder_path = self.dataset_root_folder_path + "SplitMashKLAutoEncoder/"
        assert os.path.exists(self.mash_folder_path)
        assert os.path.exists(self.split_folder_path)

        self.train_paths_list = []
        self.test_paths_list = []

        dataset_name_list = os.listdir(self.split_folder_path)

        for dataset_name in dataset_name_list:
            mash_split_folder_path = self.split_folder_path + dataset_name + "/"

            categories = os.listdir(mash_split_folder_path)

            for j, category in enumerate(categories):
                train_modelid_list_file_path = (
                    mash_split_folder_path + category + "/train.txt"
                )
                if not os.path.exists(train_modelid_list_file_path):
                    continue

                with open(train_modelid_list_file_path, "r") as f:
                    modelid_list = f.read().split()

                print("[INFO][MashDataset::__init__]")
                print(
                    "\t start load train dataset: "
                    + dataset_name
                    + "["
                    + category
                    + "], "
                    + str(j + 1)
                    + "/"
                    + str(len(categories))
                    + "..."
                )
                for modelid in tqdm(modelid_list):
                    mash_file_path = (
                        self.mash_folder_path
                        + dataset_name
                        + "/"
                        + category
                        + "/"
                        + modelid
                        + ".npy"
                    )

                    if self.preload_data:
                        mash_params = np.load(mash_file_path, allow_pickle=True).item()
                        self.train_paths_list.append(mash_params)
                    else:
                        self.train_paths_list.append(mash_file_path)

                test_modelid_list_file_path = (
                    mash_split_folder_path + category + "/val.txt"
                )
                if not os.path.exists(test_modelid_list_file_path):
                    continue

                with open(test_modelid_list_file_path, "r") as f:
                    modelid_list = f.read().split()

                print("[INFO][MashDataset::__init__]")
                print(
                    "\t start load test dataset: "
                    + dataset_name
                    + "["
                    + category
                    + "], "
                    + str(j + 1)
                    + "/"
                    + str(len(categories))
                    + "..."
                )
                for modelid in tqdm(modelid_list):
                    mash_file_path = (
                        self.mash_folder_path
                        + dataset_name
                        + "/"
                        + category
                        + "/"
                        + modelid
                        + ".npy"
                    )

                    if self.preload_data:
                        mash_params = np.load(mash_file_path, allow_pickle=True).item()
                        self.test_paths_list.append(mash_params)
                    else:
                        self.test_paths_list.append(mash_file_path)

        return

    def __len__(self):
        return len(self.train_paths_list)

    def __getitem__(self, index):
        train_index = index % len(self.train_paths_list)
        test_index = index % len(self.test_paths_list)

        np.random.seed()

        if self.preload_data:
            train_mash_params = self.train_paths_list[train_index]
        else:
            train_mash_file_path = self.train_paths_list[train_index]
            train_mash_params = np.load(train_mash_file_path, allow_pickle=True).item()

        train_rotate_vectors = train_mash_params["rotate_vectors"]
        train_positions = train_mash_params["positions"]
        train_mask_params = train_mash_params["mask_params"]
        train_sh_params = train_mash_params["sh_params"]

        train_scale_range = [0.1, 10.0]
        train_move_range = [-10.0, 10.0]

        train_random_scale = (
            train_scale_range[0] + (train_scale_range[1] - train_scale_range[0]) * np.random.rand()
        )
        train_random_translate = train_move_range[0] + (
            train_move_range[1] - train_move_range[0]
        ) * np.random.rand(3)

        train_mash_params = np.hstack(
            [
                train_rotate_vectors,
                train_positions * train_random_scale + train_random_translate,
                train_mask_params,
                train_sh_params * train_random_scale,
            ]
        )

        if self.preload_data:
            test_mash_params = self.test_paths_list[test_index]
        else:
            test_mash_file_path = self.test_paths_list[test_index]
            test_mash_params = np.load(test_mash_file_path, allow_pickle=True).item()

        test_rotate_vectors = test_mash_params["rotate_vectors"]
        test_positions = test_mash_params["positions"]
        test_mask_params = test_mash_params["mask_params"]
        test_sh_params = test_mash_params["sh_params"]


        test_mash_params = np.hstack(
            [
                test_rotate_vectors,
                test_positions,
                test_mask_params,
                test_sh_params,
            ]
        )

        train_mash_params = train_mash_params[np.random.permutation(train_mash_params.shape[0])]
        test_mash_params = test_mash_params[np.random.permutation(test_mash_params.shape[0])]

        train_mash_params = torch.tensor(train_mash_params).float()
        test_mash_params = torch.tensor(test_mash_params).float()

        feed_dict = {
            "train_input": train_mash_params,
            "train_label": train_mash_params,
            "test_input": test_mash_params,
            "test_label": test_mash_params,
        }

        return feed_dict

    def toKANDataset(self, size_max: int = -1) -> dict:
        kan_dataset = {'train_input': [],
                       'train_label': [],
                       'test_input': [],
                       'test_label': []}

        for i in range(len(self)):
            if size_max > 0 and i >= size_max:
                break

            data = self[i]

            for key in kan_dataset.keys():
                kan_dataset[key].append(data[key])

        for key, item in kan_dataset.items():
            kan_dataset[key] = torch.cat(item, dim=0)
        return kan_dataset
