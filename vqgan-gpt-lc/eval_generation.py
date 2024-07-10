import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import util.lr_sched as lr_sched
from torchvision import utils as vutils
from models.models_vq import VQModel 
#from util.utils import load_data, plot_images
import util.misc as misc
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import albumentations
import json
from torchvision import utils as vutils
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from util.misc import NativeScalerWithGradNormCount as NativeScaler
#import mlflow
import cv2
import matplotlib.pyplot as plt
from models.models_gpt import VQGANTransformer

#from metrics.fid_score.inception import InceptionV3
#from torchvision.models.inception import inception_v3
#from scipy.stats import entropy
from scipy import linalg

from common import imagenet_dict

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


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
    parser.add_argument("--distributed", default=1, type=int)
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--image_size", default=256, type=int, help="number of distributed processes")
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')

    parser.add_argument('--pkeep', type=float, default=0.8, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')
    
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--n_vision_words", default=100000, type=int)
    parser.add_argument("--n_class", default=1000, type=int)
    parser.add_argument("--tuning_codebook", type=int, default=1, help="Decoding Loss")
    parser.add_argument("--stage_1_ckpt", type=str, default="", help="Decoding Loss")
    parser.add_argument("--embed_dim", type=int, default=768, help="Decoding Loss")
    parser.add_argument("--quantizer_type", type=str, default="org", help="Decoding Loss")
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/model_16384.yaml", help="Decoding Loss")
    parser.add_argument("--use_cblinear", type=int, default=0, help="Decoding Loss")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--stage", default=2, type=int)

    parser.add_argument("--local_embedding_path", default="clip_codebook_1st_2nd_filtering.pth")
    parser.add_argument("--use_global_branch", type=int, default=0, help="Using Global Branch")
    parser.add_argument("--rate_p", default=0.1, type=float)
    parser.add_argument("--class_condition", default=1, type=int)
    parser.add_argument("--stage_2_ckpt", type=str, default="transformer_40.pt", help="Decoding Loss")

    parser.add_argument("--dataset", type=str, default="ffhq", help="")

    parser.add_argument("--top_k", default=113465, type=int)
    parser.add_argument("--gpt_type", type=str, default="small", help="")

    args = parser.parse_args()

    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    generation_save_dir = os.path.join(args.output_dir, "generation")
    os.makedirs(generation_save_dir, exist_ok=True)

    state_dict = torch.load(args.stage_2_ckpt, map_location="cpu")
    if "gpt_checkpoint_last" in args.stage_2_ckpt: #deepspeed save
        state_dict = state_dict["module"]
    #print(state_dict)
    ckpt = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model = VQGANTransformer(args).to(device=args.device)
    model.load_state_dict(ckpt, strict=True)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)
        print(model)
    model = model.module
    model.eval()
    count=0
    num_gpus = torch.cuda.device_count()
    token_freq = torch.zeros(args.n_vision_words).to(device)
    for generate_cls in range(0, np.int64(1 / num_gpus)):
        cur_cls = generate_cls * num_gpus + global_rank
        if cur_cls > 1000:
            break
        class_name = imagenet_dict[str(cur_cls)][1]
        t = os.path.join(generation_save_dir, "%s_%s.png"%(class_name, 49))
        if os.path.exists(t):
            continue
        if global_rank == 0:
            print(generate_cls, "/", np.int64(1000 / num_gpus), ":", imagenet_dict[str(cur_cls)][1])
        generate_times = np.int64(np.floor(50 / args.batch_size))
        for i in range(0, generate_times):
            num = args.batch_size

            sos_tokens = torch.ones(num, 1) * model.sos_token
            sos_tokens = sos_tokens.long().to("cuda")

            label_cls = torch.ones(num, 1) * (cur_cls + args.n_vision_words)
            label_cls = label_cls.long().to("cuda")

            if args.class_condition == 1:
                sample_indices = model.sample(None, label_cls, steps=256, top_k=args.top_k)
            else:
                sample_indices = model.sample(None, sos_tokens, steps=256, top_k=args.top_k)

            tk_index_one_hot = torch.nn.functional.one_hot(sample_indices.contiguous().view(-1), num_classes=args.n_vision_words)
            tk_index_num = torch.sum(tk_index_one_hot, dim=0)
            token_freq += tk_index_num

            x_generation = model.z_to_image(sample_indices)

            x_generation[x_generation > 1] = 1
            x_generation[x_generation < -1] = -1
            x_generation = (x_generation + 1) * 127.5

            x_generation = x_generation / 255.0

            for b in range(0, x_generation.shape[0]):
                t = os.path.join(generation_save_dir, "%s_%s.png"%(class_name, i*args.batch_size+b))
                plt.imsave(t, np.uint8(x_generation[b].detach().cpu().numpy().transpose(1, 2, 0) * 255))
    np.save(os.path.join(args.output_dir, "token_freq.npy"), np.array(token_freq.cpu().data))

    from cleanfid import fid
    fid_value = fid.compute_fid(generation_save_dir, imagenet_path + "/train", mode="clean")

    efficient_token = np.sum(np.array(token_freq.cpu().data) != 0)
    with open(os.path.join(args.output_dir, "recons.csv"), 'a') as f:
        f.write("FID, Effective_Tokens \n")
        f.write("%.4f, %d \n"%(fid_value, efficient_token))
