import os

import hiq
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import util.lr_sched as lr_sched
from torchvision import utils as vutils
from models.models_gpt import VQGANTransformer
# from models.maskgit import VQGANBidTransformer
# from util.utils import load_data, plot_images
import util.misc as misc
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import albumentations
import deepspeed
import json
from torchvision import utils as vutils
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from hiq.cv_torch import ImageLabelDataSet, get_cv_dataset, DS_PATH_IMAGENET1K, DS_PATH_DOGFOOD_3K, DS_PATH_FFHQ256_70K
from hiq.memory import total_gpu_memory_mb_nvml


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MyDSForGPT(ImageLabelDataSet):
    def __init__(self, dataset, transform=None, return_type='pair', split='train', image_size=224, convert_rgb=True,
                 img_key=None, max_num=100000):
        super().__init__(dataset, transform, return_type, split, image_size, convert_rgb, img_key, max_num=max_num)
        self.rescaler = albumentations.SmallestMaxSize(max_size=image_size)
        self.cropper = albumentations.RandomCrop(height=image_size, width=image_size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item[self.img_key]
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = np.array(img).astype(np.uint8)
        image = self.preprocessor(image=img)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        label = item[self.label_key] if self.label_key in item else -1
        return [-1, image, label]


def configure_optimizers(model, args):
    decay, no_decay = set(), set()
    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    for mn, m in model.transformer.named_modules():
        # print(m)
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
    no_decay.add("pos_emb")
    param_dict = {pn: p for pn, p in model.transformer.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95))
    return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument("--distributed", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--image_size", default=256, type=int, help="number of distributed processes")
    parser.add_argument('--lr', type=float, default=4.5e-04, help='Commitment loss scalar.')
    parser.add_argument(
        "--min_lr", type=float, default=1e-05, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )
    parser.add_argument('--latent-dim', type=int, default=1024, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--imagenet_path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=2, help='Input batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.,
                        help='Weighting factor for perceptual loss.')
    parser.add_argument("--warmup_iterators", type=int, default=5000, metavar="N", help="epochs to warmup LR")
    parser.add_argument("--keeplr_iterators", type=int, default=80000, metavar="N", help="epochs to warmup LR")
    parser.add_argument("--max_iterators", type=int, default=450000, metavar="N", help="epochs to warmup LR")
    parser.add_argument('--pkeep', type=float, default=0.8, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--n_vision_words", default=32000, type=int)
    parser.add_argument("--n_class", default=1000, type=int)
    parser.add_argument("--tuning_codebook", type=int, default=1, help="Decoding Loss")
    parser.add_argument("--clip_embedding_path", default="clip_codebook_1st_2nd_filtering.pth")
    parser.add_argument("--stage_1_ckpt", type=str, default="stage_1_llama_tuned-40.pth", help="Decoding Loss")
    parser.add_argument("--embed_dim", type=int, default=768, help="Decoding Loss")
    parser.add_argument("--quantizer_type", type=str, default="org", help="Decoding Loss")
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/model_16384.yaml", help="Decoding Loss")
    parser.add_argument("--use_cblinear", type=int, default=0, help="Decoding Loss")
    parser.add_argument("--use_crossatt_dec", type=int, default=0, help="Decoding Loss")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--stage", default=2, type=int)
    parser.add_argument("--local_embedding_path", default="cluster_codebook_1000cls_100000.pth", type=str,
                        help="path of llama model")
    parser.add_argument("--rate_p", default=0.1, type=float)
    parser.add_argument("--class_condition", default=1, type=int)
    parser.add_argument("--dataset", type=str, default="imagenet", help="")
    parser.add_argument("--gpt_type", type=str, default="small", help="")
    parser.add_argument("--label_smooth", default=0, type=int)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    repo_id = "wufuheng/VQ"
    if not os.path.exists(args.local_embedding_path):
        hiq.ensure_folder(args.local_embedding_path)
        hf_hub_download(repo_id, args.local_embedding_path)
    if not os.path.exists(args.stage_1_ckpt):
        hiq.ensure_folder(args.stage_1_ckpt)
        hf_hub_download(repo_id, args.stage_1_ckpt)

    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    dl = get_cv_dataset(path=DS_PATH_DOGFOOD_3K if args.dataset == "imagenet" else DS_PATH_FFHQ256_70K,  # DS_PATH_IMAGENET1K
                        image_size=args.image_size,
                        batch_size=args.batch_size,
                        return_loader=False,
                        max_num=500,
                        datasetclass=MyDSForGPT)
    dataset_train, dataset_val = dl['train'], dl['test'] if 'test' in dl else None
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True
    )
    if dataset_val:
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=False
        )

    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    model = VQGANTransformer(args).to(device=args.device)
    optimizer = configure_optimizers(model, args)
    # for param in model.vqgan.parameters():
    #    param.requires_grad = False
    model_engine = model
    for name, param in model.vqgan.named_parameters():
        param.data = param.data.float()
        param.requires_grad = False

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    start_epoch = 0
    p = os.path.join(args.output_dir, "gpt_checkpoint_last")
    if os.path.exists(p):
        checkpoint = torch.load(p, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loading Checkpoint {str(p)} from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        print(f"epoch: {epoch}, GPU: {total_gpu_memory_mb_nvml() // 1024}GB")
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)

        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Training Epoch: [{}]".format(epoch)
        print_freq = 10
        metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        for i, [image_ids, images, label_cls] in enumerate(
                metric_logger.log_every(data_loader_train, print_freq, header)):
            cur_iter = len(data_loader_train) * epoch + i
            if cur_iter >= args.max_iterators:
                break
            imgs = images.to(device=args.device)
            if label_cls[0]==-1:
                label_cls = label_cls.unsqueeze(-1).to(device) + args.n_vision_words
                loss = model_engine(imgs, label_cls)
            else:
                loss = model(imgs)
            loss.backward()
            optimizer.step()
            lr_sched.adjust_learning_rate_gpt(optimizer, cur_iter, args)
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value = loss.item()
            metric_logger.update(loss=loss_value)
            misc.all_reduce_mean(loss_value)
            loss_value_reduce = misc.all_reduce_mean(loss_value)

            if log_writer is not None and cur_iter % 1000 == 0:
                epoch_1000x = int(cur_iter)
                log_writer.add_scalar("Iter/Loss", loss_value_reduce, epoch_1000x)

        metric_logger.synchronize_between_processes()
        print("Training Averaged stats:", metric_logger)

        if log_writer is not None:
            log_writer.add_scalar("Epoch/Loss", loss_value_reduce, epoch)

        if dataset_val is not None and epoch % 5 == 0:
            header = "Validation Epoch: [{}]".format(epoch)
            for i, [image_ids, images, label_cls] in enumerate(
                    metric_logger.log_every(data_loader_val, print_freq, header)):
                cur_iter = len(data_loader_train) * epoch + i
                imgs = images.to(device=args.device)
                label_cls = label_cls.unsqueeze(-1).to(device) + args.n_vision_words
                with torch.no_grad():
                    if args.class_condition == 1:
                        loss = model_engine(imgs, label_cls)
                    else:
                        loss = model(imgs)
                loss_value = loss.item()
                metric_logger.update(val_loss=loss_value)
                misc.all_reduce_mean(loss_value)
                loss_value_reduce = misc.all_reduce_mean(loss_value)

            if log_writer is not None:
                log_writer.add_scalar("Epoch/Val_Loss", loss_value_reduce, epoch)

            metric_logger.synchronize_between_processes()
            print("Validation Averaged stats:", metric_logger)

        client_sd = {"epoch": epoch, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict()}
        p = os.path.join(args.output_dir, "gpt_checkpoint_last")
        torch.save(client_sd, p)
        print(f"Save model to {p}")

        if epoch % 10 == 0 or cur_iter == args.max_iterators:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"transformer_{epoch}.pt"))
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch  # ,
        }
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if cur_iter >= args.max_iterators:
            break
