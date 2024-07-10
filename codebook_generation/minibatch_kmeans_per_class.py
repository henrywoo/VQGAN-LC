from sklearn.cluster import MiniBatchKMeans
import numpy as np
import torch
from kmeans_pytorch import kmeans, kmeans_predict
import os
from hiq.cv_torch import IN_CAT
import argparse
parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=1000, type=int)
parser.add_argument("--n_class", default=1000, type=int)
parser.add_argument("--k", default=100, type=int)
parser.add_argument("--downsample", default=4, type=int)
parser.add_argument("--imagenet_feature_path", default="", type=str)
parser.add_argument("--save_dir", default="clustering_centers", type=str)
args = parser.parse_args()

save_path = args.save_dir
os.makedirs(save_path, exist_ok=True)
if args.n_class == 300:
    select_classes = np.load("select_300_class.npy")
    print("300 Selected Classes")
else:
    select_classes = np.arange(0, args.n_class)
    print("The first %d classes"%(args.n_class))
count = args.start
k=args.k
#for i in range(0, 1000):
for i in select_classes[args.start:args.end]:
    class_label = i
    count = count + 1
    t = os.path.join(save_path, "class_center_%d_%d.npy"%(count, class_label))
    if os.path.exists(t):
        continue
    print(count, ", Processing:", class_label, "Loading")
    t = IN_CAT[class_label]
    t = '/'.join(t)
    dir_path = os.path.join(args.imagenet_feature_path, t)
    files = os.listdir(dir_path)
    #features = []
    #for file in files:
    #    features.append(np.load(os.path.join(dir_path, file)))
    features = [torch.from_numpy(np.load(os.path.join(dir_path, file))) for file in files]
    features = torch.cat(features, dim=0)
    #features = features.view(-1, 16, 16, 768)[:, ::4, ::4, :]
    features = features.view(-1, 16, 16, 768).contiguous().permute(0, 3, 1, 2)
    features = torch.nn.AvgPool2d((args.downsample, args.downsample))(features.float())
    features = features.contiguous().permute(0, 2, 3, 1)
    print(features.shape)

    print(count, ", Processing:", class_label, "Clustering")
    features = features.reshape(-1, 768).permute(1, 0)
    #x = torch.from_numpy(features)
    label, center  = kmeans(X=features, num_clusters=k, device=torch.device('cuda:0'))
    np.save(os.path.join(save_path, "class_center_%d_%d.npy"%(count, class_label)), center.data)



dir_path = args.save_dir
files = os.listdir(args.save_dir)

features = [torch.from_numpy(np.load(os.path.join(dir_path, file))) for file in files]
features = torch.cat(features, dim=0)
torch.save(features, "clustering_codebook_imagenet1k_100000.pth")
