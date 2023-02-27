import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from get_dataset import get_dataset
from config import NUM_CLASSES


def get_transform(image_size, random_crop=False, random_horizontal_flip=False, random_vertical_flip=False,
                       gaussian_blur=False, random_rotation=False,
                       normalize_mean=(0.5,), normalize_std=(0.5,)):
    transform_list = [transforms.Resize((image_size, image_size))]
    if random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if random_crop:
        transform_list.append(transforms.RandomCrop(image_size, padding=(4 if image_size == 32 else 8)))
    if random_vertical_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if gaussian_blur:
        transform_list.append(transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)))
    if random_rotation:
        transform_list.append(transforms.RandomRotation(degrees=(30, 70)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(normalize_mean, normalize_std))
    return transforms.Compose(transform_list)        


def get_meta_train_dataloader(name, image_size, batch_size, default_data_path, 
        aug=False, split=0, total_split=10, num_workers=1):

    if name not in ["tiny_imagenet"]:
        raise NotImplementedError
        
    # Get split information
    split_info = np.load(f"{default_data_path}/{name}/{total_split}_split/split_{split}.npz")
    label_list = list(split_info["label_list"])
    idx_train = list(split_info["idx_train"])
    idx_valid = list(split_info["idx_valid"])

    def target_transform(y):
        return label_list.index(y)

    if aug:
        transform_train = get_transform(image_size, random_crop=True, random_horizontal_flip=True,
                                        random_vertical_flip=True, gaussian_blur=True, random_rotation=True)
        transform_test = get_transform(image_size)
    else:
        transform_train = get_transform(image_size, random_crop=True, random_horizontal_flip=True)
        transform_test = get_transform(image_size)
        

    train_ds = get_dataset(name, train=True, default_data_path=default_data_path,
                transform=transform_train, target_transform=target_transform)
    valid_ds = get_dataset(name, train=True, default_data_path=default_data_path,
                transform=transform_test, target_transform=target_transform)

    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": False, "drop_last": True}
    train_loader = DataLoader(train_ds, sampler=SubsetRandomSampler(idx_train), **kwargs)
    valid_loader = DataLoader(valid_ds, sampler=SubsetRandomSampler(idx_valid), **kwargs)
    test_loader = None

    return train_loader, valid_loader, test_loader, len(label_list)


def get_meta_test_dataloader(name, image_size, batch_size, default_data_path, split=None, num_workers=1, aug=False):

    if name in ['quickdraw']:
        num_instances = 20
    else:
        num_instances = None

    if aug:
        transform_train = get_transform(image_size, random_crop=True, random_horizontal_flip=True,
                                        random_vertical_flip=True, gaussian_blur=True, random_rotation=True)
        transform_test = get_transform(image_size)
    else:
        transform_train = get_transform(image_size, random_crop=True, random_horizontal_flip=True)
        transform_test = get_transform(image_size)

    train_ds = get_dataset(name, train=True, default_data_path=default_data_path,
                transform=transform_train)
    test_ds = get_dataset(name, train=False, default_data_path=default_data_path,
                transform=transform_test)

    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True, "drop_last": True}
    if num_instances:
        train_idx = []
        for c in range(NUM_CLASSES[name]):
            try:
                train_idx.extend(list(np.argwhere(train_ds.labels == c)[:num_instances, 0]))
            except AttributeError: 
            #    print('error')
                train_idx.extend(list(np.argwhere(np.array(train_ds.targets) == c)[:50, 0]))
        train_loader = DataLoader(train_ds, sampler=SubsetRandomSampler(train_idx), **kwargs)
        test_loader = DataLoader(test_ds, **kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
        test_loader = DataLoader(test_ds, **kwargs)
    return train_loader, test_loader, None, NUM_CLASSES[name]


def get_dataloader(mode, default_data_path, image_size, batch_size, ds_name, ds_split, aug=False):
    if mode in ['meta_train', 'meta_valid']:
        train_loader, valid_loader, _, n_classes = get_meta_train_dataloader(
            name=ds_name,
            image_size=image_size,
            batch_size=batch_size,
            default_data_path=default_data_path,
            split=ds_split,
            aug=aug)
    elif mode == 'meta_test':
        train_loader, valid_loader, _, n_classes = get_meta_test_dataloader(
            name=ds_name,
            image_size=image_size,
            batch_size=batch_size,
            default_data_path=default_data_path,
            split=ds_split,
            aug=aug)
    else: raise NotImplementedError
    return train_loader, valid_loader, n_classes

def get_cross_domain_dataloader(name, image_size, batch_size, aug=False):
    train_datamgr = SimpleDataManager(ds_name=name, image_size=image_size, batch_size=batch_size, train=True)
    train_loader = train_datamgr.get_data_loader(aug=False)
    test_datamgr = SimpleDataManager(ds_name=name, image_size=image_size, batch_size=batch_size, train=False)
    test_loader = test_datamgr.get_data_loader(aug=False)
    n_classes = NUM_CLASSES[name]
    return train_loader, test_loader, n_classes