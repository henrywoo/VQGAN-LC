import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import util.misc as misc
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

##############
import clip_encoder as clip
from PIL import Image
import yaml
import torch
from omegaconf import OmegaConf
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
from hiq.cv_torch import IN_CAT

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config


def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

class ImageNetDataset(Dataset):
    def __init__(self, data_root, image_size, max_words=30, n_class=1000, partition="train", device="cpu"):

        self.max_words = max_words
        self.device = device
        self.image_size = image_size

        self.data_root = data_root

        u, self.preprocess = clip.load("ViT-L/14", device=DEVICE)

        self.image_ids = []
        self.class_labels = []
        with open("imagenet_split/" + partition + "/class_labels.txt") as f:
            for line in f.readlines():
                image_id, class_label = line.strip('\n').split(",")
                if int(class_label) < n_class: #only select 100 class
                    ##
                    if partition == "train":
                        self.image_ids.append(image_id)
                    elif partition == "val":
                        self.image_ids.append(image_id)
                    self.class_labels.append(int(class_label))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):

        image_ids = self.image_ids[index]
        #image = Image.open(os.path.join(self.data_root, image_ids))
        image = Image.open(os.path.join(self.data_root, image_ids))

        image = self.preprocess(image)

        return [image, image_ids]

def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    # Model parameters
    parser.add_argument("--model", default="llama7B", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--max_seq_len", type=int, default=512, metavar="LENGTH", help="the maximum sequence length")
    parser.add_argument("--image_size", type=int, default=256, help="Decoding Loss")
    parser.add_argument("--n_class", default=1000, type=int)    
    # Dataset parameters
    parser.add_argument("--data_path", default="instruction_dataset/", type=str, help="dataset path")
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--imagenet_path", default="", type=str, help="path of llama model")
    parser.add_argument("--save_path", default="Imagenet_clip_features", type=str, help="path of llama model")
    return parser

from hiq.cv_torch import *
from hiq import ensure_folder, print_model
def main(args):

    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    model, preprocess_ = clip.load("ViT-L/14", device=DEVICE)
    dl = get_cv_dataset(path=DS_PATH_IMAGENET1K,
                        transform=preprocess_,
                        image_size=args.image_size,
                        batch_size=args.batch_size,
                        return_loader=False)
    dataset_train, dataset_val = dl['train'], dl['validation']

    '''dataset_train = ImageNetDataset(
        data_root=args.imagenet_path, image_size=args.image_size, max_words=args.max_seq_len, n_class=args.n_class, partition="train", device=device
    )
    dataset_val = ImageNetDataset(
        data_root=args.imagenet_path, image_size=args.image_size, max_words=args.max_seq_len, n_class=args.n_class, partition="val", device=device
    )'''

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True
        )

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True
        )

        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    #config = load_config(args.vq_config_path, display=True)
    ##model, _ = clip.load("ViT-L/14", device=DEVICE)
    model.to(device)


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = ""
    print_freq = 10
    
    #os.makedirs(args.save_path, exist_ok=True)

    model.eval()
    print_model(model)
    count = {}
    for i in range(1000):
        count[i] = 0
    for data_iter_step, [images, image_id] in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        images = images.to(device)
        with torch.no_grad():
            _, z_flattened = model.encode_image(images)

        sz = z_flattened.shape[0]
        for j in range(0, sz):
            class_ = image_id[j].item()
            #save_dir = "/".join(image_id[j].split("/")[:-1])
            t = IN_CAT[class_]
            t.append(f"{count[class_]}.npz")
            t.insert(0, args.save_path)
            pth = "/".join(t)
            ensure_folder(pth)
            #os.makedirs(os.path.join(args.save_path, save_dir), exist_ok=True)
            np.save(pth, np.array(z_flattened[j].cpu().data))
            count[class_] += 1
        

if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
