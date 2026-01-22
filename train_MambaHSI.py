import os
import pdb

os.environ['CUDA_VISIBLE_DEVICES']='1'
import time
import torch
import random
import argparse
import numpy as np
from torchvision import models,transforms
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss,resize
from utils.evaluation import Evaluator
from utils.HSICommonUtils import normlize3D, ImageStretching

# import matplotlib.pyplot as plt
# from visual.visualize_map import DrawResult
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict, visualize_cluster_centers
from PIL import Image
from model.MambaHSI import *
from model.mHC_HSI import mHC_MambaHSI_Integrated
#from model.resnet_connection import ImageHyperConnectionTransformer
#from model.manifold_connection_mhc import ImageHyperConnectionTransformer
#from model.mHC_spa_spe import ImageHyperConnectionTransformer
#from model.mHC_cluster import ImageHyperConnectionTransformer
from model.physical_mHC import ImageHyperConnectionTransformer
from calflops import calculate_flops
from collections import Counter
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())
import cv2

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from matplotlib.colors import ListedColormap, BoundaryNorm

def allocate_subclusters_proportional(counts, total_subclusters):
    """Allocate number of subsets per class, proportional to class pixel count."""
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    alloc = (counts / total) * total_subclusters
    alloc = np.maximum(1, np.floor(alloc)).astype(int)

    diff = total_subclusters - alloc.sum()
    if diff > 0:
        frac = alloc - np.floor(alloc)
        for i in np.argsort(-frac):
            if diff <= 0:
                break
            alloc[i] += 1
            diff -= 1
    elif diff < 0:
        for i in np.argsort(counts):
            if diff >= 0:
                break
            if alloc[i] > 1:
                alloc[i] -= 1
                diff += 1
    return alloc

def split_label_map_by_spectra(hsi, label_map, num_classes, n_subclasses=100, n_pca_components=20, random_state=42):
    """
    Split each class (1–16) into subsets using spectral clustering.
    Background (0) is ignored.
    Output labels are 0..n_subclasses-1 globally.
    """
    H, W, B = hsi.shape
    hsi = hsi.astype(np.float32)
    for i in range(B):
        hsi[:, :, i] = (hsi[:, :, i] - np.min(hsi[:, :, i])) / (np.max(hsi[:, :, i]) - np.min(hsi[:, :, i]) + 1e-10)
    new_label_map = np.full((H, W), fill_value=-1, dtype=int)

    # Consider only labels 1–16
    class_labels = np.arange(0, num_classes)
    counts = [np.sum(label_map == c) for c in class_labels]
    allocs = allocate_subclusters_proportional(counts, n_subclasses)

    # Fit global PCA and normalization on all non-background pixels
    mask_valid = label_map >= 0
    X_all = hsi[mask_valid].reshape(-1, B)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=min(n_pca_components, B), random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # To retrieve spectral embeddings per pixel
    coords_valid = np.argwhere(mask_valid)
    flat_indices_valid = np.ravel_multi_index((coords_valid[:,0], coords_valid[:,1]), (H,W))

    next_label = 0
    mapping = {}

    for i, c in enumerate(class_labels):
        n_sub = int(allocs[i])
        mask = (label_map == c)
        idx_flat = np.nonzero(mask.ravel())[0]
        if len(idx_flat) == 0:
            continue

        # get corresponding indices in X_pca
        valid_mask = np.isin(flat_indices_valid, idx_flat)
        Xc = X_pca[valid_mask]

        if n_sub == 1:
            sub_labels = np.zeros(len(Xc), dtype=int)
        else:
            n_sub = min(n_sub, len(Xc))
            km = KMeans(n_clusters=n_sub, random_state=random_state, n_init=5)
            sub_labels = km.fit_predict(Xc)

        # assign unique global labels
        global_labels = np.array([next_label + l for l in sub_labels], dtype=int)
        new_label_map.ravel()[idx_flat] = global_labels
        mapping[c] = list(range(next_label, next_label + n_sub))
        next_label += n_sub

    return new_label_map, mapping


def seeded_kmeans_hsi(hsi, reassigned_label_map):
    """
    Cluster HSI pixels using seeded KMeans initialized with mean spectra
    of the 100 subsets from reassigned label map.

    Args:
        hsi: ndarray (H, W, B)
        reassigned_label_map: ndarray (H, W), values 0..99 (100 subsets)

    Returns:
        new_label_map: ndarray (H, W), cluster IDs 0..99
        mean_spectra: ndarray (100, B), mean spectra of each subset (used as initial centers)
    """
    H, W, B = hsi.shape
    pixels = hsi.reshape(-1, B)

    # 1. Compute mean spectra for each subset
    subset_labels = np.unique(reassigned_label_map)
    subset_labels = subset_labels[subset_labels >= 0]  # ignore background if any
    mean_spectra = []
    for s in subset_labels:
        mask = (reassigned_label_map == s).ravel()
        subset_pixels = pixels[mask]
        mean_spec = subset_pixels.mean(axis=0)
        mean_spectra.append(mean_spec)
    mean_spectra = np.stack(mean_spectra, axis=0)  # shape (100, B)

    # 2. Run seeded KMeans with these mean vectors
    km = KMeans(n_clusters=len(mean_spectra), init=mean_spectra, n_init=1, random_state=42)
    cluster_labels = km.fit_predict(pixels)  # shape (H*W,)

    # 3. Reshape back to H x W
    new_label_map = cluster_labels.reshape(H, W)

    return new_label_map, mean_spectra




def kmeans_clustering(image, n_clusters=19, random_state=42):
    """
    Perform K-means clustering on an image with shape (H, W, C)

    Args:
        image: numpy array of shape (310, 640, 103)
        n_clusters: number of clusters (default: 19)
        random_state: random seed for reproducibility

    Returns:
        cluster_map: array of shape (310, 640) with cluster labels
        kmeans_model: trained KMeans model
    """
    # Reshape image to 2D array: (H*W, C)
    H, W, C = image.shape
    pixels = image.reshape(-1, C)  # Shape: (310*640, 103) = (198400, 103)

    print(f"Original image shape: {image.shape}")
    print(f"Reshaped pixels shape: {pixels.shape}")
    print(f"Number of clusters: {n_clusters}")

    # Standardize the features (important for K-means)
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(pixels_scaled)

    # Reshape back to original spatial dimensions
    cluster_map = cluster_labels.reshape(H, W)

    return cluster_map, kmeans, scaler

def supervised_kmeans_from_label(image, label_map, n_classes=9, random_state=42):
    """
    Supervised K-Means using class mean vectors (computed from labeled pixels in the image)
    as initialization centers.

    Args:
        image: np.ndarray, shape (H, W, C)
            Hyperspectral image cube.
        label_map: np.ndarray, shape (H, W)
            Label map, where 0..n_classes-1 indicate class labels.
            Pixels with label < 0 or 255 can be treated as unlabeled (optional).
        n_classes: int
            Number of known classes.
        random_state: int
            Random seed for reproducibility.

    Returns:
        cluster_map: np.ndarray, shape (H, W)
            Cluster assignment map after supervised K-means.
        kmeans: trained sklearn KMeans model
        scaler: StandardScaler object
    """

    H, W, C = image.shape
    pixels = image.reshape(-1, C)
    labels_flat = label_map.reshape(-1)

    print(f"Image shape: {image.shape}")
    print(f"Unique labels in label_map: {np.unique(labels_flat)}")

    # Optional: mask out unlabeled pixels (e.g., label = -1 or 255)
    valid_mask = (labels_flat >= 0) & (labels_flat < n_classes)
    labeled_pixels = pixels[valid_mask]
    labeled_labels = labels_flat[valid_mask]

    # --- Standardize using only labeled pixels (to simulate training normalization) ---
    scaler = StandardScaler()
    scaler.fit(labeled_pixels)
    pixels_scaled = scaler.transform(pixels)
    labeled_scaled = scaler.transform(labeled_pixels)

    # --- Compute class mean vectors ---
    class_means = []
    for cls in range(n_classes):
        cls_vectors = labeled_scaled[labeled_labels == cls]
        if cls_vectors.shape[0] == 0:
            raise ValueError(f"No labeled pixels for class {cls}")
        class_mean = cls_vectors.mean(axis=0)
        class_means.append(class_mean)
    init_centers = np.stack(class_means, axis=0)
    print(f"Initialized {n_classes} class centers from labeled pixels")

    # --- K-means initialized from class means ---
    kmeans = KMeans(
        n_clusters=n_classes,
        init=init_centers,
        n_init=1,
        random_state=random_state
    )
    cluster_labels = kmeans.fit_predict(pixels_scaled)
    cluster_map = cluster_labels.reshape(H, W)

    return cluster_map, kmeans, scaler


def supervised_subkmeans_from_label(
        image,
        label_map,
        n_classes=9,
        sub_clusters_per_class=3,
        random_state=42
):
    """
    Perform supervised K-means with sub-clusters per class.
    Example: 9 classes × 3 sub-clusters = 27 total clusters.

    Args:
        image: np.ndarray, shape (H, W, C)
            Hyperspectral image cube.
        label_map: np.ndarray, shape (H, W)
            Label map with integer class labels (0..n_classes-1).
            Unlabeled pixels can be marked as -1 or 255.
        n_classes: int
            Number of known classes.
        sub_clusters_per_class: int
            Number of sub-clusters per class.
        random_state: int
            Random seed for reproducibility.

    Returns:
        cluster_map: np.ndarray, shape (H, W)
            Final cluster assignments (0..n_classes*sub_clusters_per_class-1).
        kmeans: trained global KMeans model
        scaler: StandardScaler object
    """
    H, W, C = image.shape
    pixels = image.reshape(-1, C)
    labels_flat = label_map.reshape(-1)

    print(f"Image shape: {image.shape}")
    print(f"Unique labels: {np.unique(labels_flat)}")

    # Mask valid labeled pixels
    valid_mask = (labels_flat >= 0) & (labels_flat < n_classes)
    labeled_pixels = pixels[valid_mask]
    labeled_labels = labels_flat[valid_mask]

    # --- Standardize on labeled data ---
    scaler = StandardScaler()
    scaler.fit(labeled_pixels)
    pixels_scaled = scaler.transform(pixels)
    labeled_scaled = scaler.transform(labeled_pixels)

    # --- Step 1: Within each class, perform local sub-clustering ---
    all_centers = []
    for cls in range(n_classes):
        cls_vectors = labeled_scaled[labeled_labels == cls]
        if cls_vectors.shape[0] < sub_clusters_per_class:
            raise ValueError(f"Not enough pixels for class {cls} to form {sub_clusters_per_class} sub-clusters.")

        kmeans_local = KMeans(
            n_clusters=sub_clusters_per_class,
            random_state=random_state + cls,  # different seed per class
            n_init=10
        )
        kmeans_local.fit(cls_vectors)
        all_centers.append(kmeans_local.cluster_centers_)
        print(f"Class {cls}: {sub_clusters_per_class} sub-centers computed.")

    # Stack all sub-centers as initialization for global K-means
    init_centers = np.vstack(all_centers)
    total_clusters = init_centers.shape[0]
    print(f"Total clusters: {total_clusters} ({n_classes} × {sub_clusters_per_class})")

    # --- Step 2: Global clustering initialized by sub-cluster centers ---
    kmeans_global = KMeans(
        n_clusters=total_clusters,
        init=init_centers,
        n_init=1,  # use our centers directly
        random_state=random_state
    )
    cluster_labels = kmeans_global.fit_predict(pixels_scaled)
    cluster_map = cluster_labels.reshape(H, W)

    return cluster_map, kmeans_global, scaler


def cluster_map(data1, data2, data3):

    # ----------------------------------------------------------
    # Replace these with your real 145×145 datasets
    # ----------------------------------------------------------
    datasets = [data1, data2, data3]

    # ----------------------------------------------------------
    # Create a 101-color discrete version of gist_rainbow
    # ----------------------------------------------------------
    base = plt.cm.gist_rainbow
    cmap_discrete1 = ListedColormap(base(np.linspace(0, 1, 30)))
    cmap_discrete2 = ListedColormap(base(np.linspace(0, 1, 50)))
    cmap_discrete3 = ListedColormap(base(np.linspace(0, 1, 100)))
    cmap_color = [cmap_discrete1, cmap_discrete2, cmap_discrete3]

    # Discrete boundaries: 0,1,2,...,100,101
    bounds1 = np.arange(0, 31)
    norm1 = BoundaryNorm(bounds1, cmap_discrete1.N)

    bounds2 = np.arange(0, 51)
    norm2 = BoundaryNorm(bounds2, cmap_discrete2.N)

    bounds3 = np.arange(0, 101)
    norm3 = BoundaryNorm(bounds3, cmap_discrete3.N)
    norm = [norm1, norm2, norm3]

    titles = ["Cluster 1", "Cluster 2", "Cluster 3"]

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    plt.figure(figsize=(15, 5))

    for i, arr in enumerate(datasets, 1):
        ax = plt.subplot(1, 3, i)

        im = ax.imshow(arr, cmap=cmap_color[i-1], norm=norm[i-1])
        ax.set_title(titles[i - 1])
        ax.set_xticks([])
        ax.set_yticks([])

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if i == 1:
            cbar.set_ticks([1, 15, 30])
            cbar.set_ticklabels(["1", "15", "30"])
        if i == 2:
            cbar.set_ticks([1, 25, 50])
            cbar.set_ticklabels(["1", "25", "50"])
        if i == 3:
            cbar.set_ticks([1, 50, 100])
            cbar.set_ticklabels(["1", "50", "100"])

    plt.tight_layout()
    plt.savefig("cluster_map_her.pdf", dpi=300, bbox_inches="tight")


def visualize_clusters(original_image, cluster_map, n_clusters=19):
    """
    Visualize the original image and clustering results
    """
    # Create a colormap for the clusters
    colors = plt.cm.get_cmap('tab20', n_clusters)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot original image (first 3 channels as RGB if available)
    if original_image.shape[2] >= 3:
        axes[0, 0].imshow(original_image[:, :, :3])
        axes[0, 0].set_title('Original Image (First 3 Channels)')
    else:
        axes[0, 0].imshow(original_image[:, :, 0], cmap='gray')
        axes[0, 0].set_title('Original Image (First Channel)')
    axes[0, 0].axis('off')

    # Plot cluster map
    im = axes[0, 1].imshow(cluster_map, cmap=colors)
    axes[0, 1].set_title(f'K-means Clustering ({n_clusters} clusters)')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # Plot cluster histogram
    cluster_counts = np.bincount(cluster_map.flatten())
    cluster_ids = np.arange(n_clusters)

    axes[1, 0].bar(cluster_ids, cluster_counts, color=colors(cluster_ids))
    axes[1, 0].set_xlabel('Cluster ID')
    axes[1, 0].set_ylabel('Pixel Count')
    axes[1, 0].set_title('Cluster Size Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Add count labels on bars
    for i, count in enumerate(cluster_counts):
        axes[1, 0].text(i, count + max(cluster_counts) * 0.01, f'{count:,}',
                        ha='center', va='bottom', fontsize=8)

    # Plot sorted cluster sizes (head-tail distribution)
    sorted_indices = np.argsort(cluster_counts)[::-1]
    sorted_counts = cluster_counts[sorted_indices]

    axes[1, 1].bar(range(n_clusters), sorted_counts, color=colors(sorted_indices))
    axes[1, 1].set_xlabel('Cluster Rank (by size)')
    axes[1, 1].set_ylabel('Pixel Count')
    axes[1, 1].set_title('Cluster Sizes (Sorted - Head-Tail Distribution)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("Kmeans.png")

    return sorted_indices, sorted_counts


def analyze_cluster_statistics(cluster_map, n_clusters=19):
    """
    Analyze and print detailed statistics about the clusters
    """
    cluster_counts = np.bincount(cluster_map.flatten())
    total_pixels = cluster_map.size

    print("=" * 60)
    print("CLUSTER STATISTICS")
    print("=" * 60)
    print(f"Total pixels: {total_pixels:,}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Image dimensions: {cluster_map.shape}")
    print("\nCluster distribution:")
    print("-" * 40)

    # Sort clusters by size (descending)
    sorted_indices = np.argsort(cluster_counts)[::-1]

    for rank, cluster_id in enumerate(sorted_indices):
        count = cluster_counts[cluster_id]
        percentage = (count / total_pixels) * 100
        cumulative_pct = (cluster_counts[sorted_indices[:rank + 1]].sum() / total_pixels) * 100

        print(f"Rank {rank + 1:2d}: Cluster {cluster_id:2d} - {count:>8,} pixels "
              f"({percentage:6.2f}%) - Cumulative: {cumulative_pct:6.2f}%")

    # Head-tail analysis
    head_threshold = 0.8  # 80% of pixels
    cumulative = 0
    head_clusters = []
    tail_clusters = []

    for cluster_id in sorted_indices:
        cumulative += cluster_counts[cluster_id] / total_pixels
        if cumulative <= head_threshold:
            head_clusters.append(cluster_id)
        else:
            tail_clusters.append(cluster_id)

    print(f"\nHead-Tail Analysis (80% threshold):")
    print(f"Head clusters ({len(head_clusters)}): {head_clusters}")
    print(f"Tail clusters ({len(tail_clusters)}): {tail_clusters}")
    print(f"Head covers {len(head_clusters) / n_clusters * 100:.1f}% of clusters "
          f"with {head_threshold * 100:.1f}% of pixels")

    return sorted_indices, head_clusters, tail_clusters



def vis_a_image(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=False):
    visualize_predict(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=only_vis_label)
    visualize_predict(gt_vis,pred_vis,save_single_predict_path.replace('.png','_mask.png'),save_single_gt_path,only_vis_label=True)


# random seed setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int,default=0)
    parser.add_argument('--data_set_path',type=str,default='./data')
    parser.add_argument('--work_dir',type=str,default='./')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_samples', type=int, default=30)
    parser.add_argument('--val_samples', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='RUNS')
    parser.add_argument('--record_computecost',type=bool,default=True)

    args = parser.parse_args()
    return args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_parser()
record_computecost = args.record_computecost
exp_name = args.exp_name
#seed_list = [0,1,2,3,4,5,6,7,8,9]  #
seed_list = [0]  #

num_list = [args.train_samples, args.val_samples]

dataset_index = args.dataset_index

max_epoch = args.max_epoch
learning_rate = args.lr

net_name = 'MambaHSI'

paras_dict = {'net_name':net_name,'dataset_index':dataset_index,'num_list':num_list,
              'lr':learning_rate,'seed_list':seed_list}


                      # 0        1         2         3        4     5
data_set_name_list = ['IN', 'UP', 'HanChuan', 'HongHu', 'Houston', 'LN', 'SA']
data_set_name = data_set_name_list[dataset_index]

if data_set_name in ['HanChuan','Houston', 'LN']:
    split_image = True
else:
    split_image = False

transform = transforms.Compose([
    # transforms.Resize((2048, 1024)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[123.6750, 116.2800, 103.5300], std=[58.395, 57.120, 57.3750]),
])


if __name__ == '__main__':
    data_set_path = args.data_set_path
    work_dir = args.work_dir
    setting_name = 'tr{}val{}'.format(str(args.train_samples),str(args.val_samples)) + '_lr{}'.format(str(learning_rate))

    dataset_name = data_set_name

    exp_name = args.exp_name

    save_folder = os.path.join(work_dir, exp_name, net_name, dataset_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("makedirs {}".format(save_folder))

    save_log_path = os.path.join(save_folder,'train_tr{}_val{}.log'.format(num_list[0],num_list[1]))
    logger = setup_logger(name='{}'.format(dataset_name),logfile=save_log_path)
    torch.cuda.empty_cache()

    logger.info(save_folder)

    data, gt = data_load_operate.load_data(data_set_name, data_set_path)
    print("Creating sample image for demonstration...")
    num_clusters = 10

    height, width, channels = data.shape

    gt_reshape = gt.reshape(-1)
    height, width, channels = data.shape
    img = ImageStretching(data)
    #img = (data - np.min(data) / (np.max(data) - np.min(data)))
    # img = np.zeros_like(data)
    # for i in range(data.shape[-1]):
    #     input_max = np.max(data[:, :, i])
    #     input_min = np.min(data[:, :, i])
    #     img[:, :, i] = ((data[:, :, i] - input_min) / (input_max - input_min))

    class_count = max(np.unique(gt))

    flag_list = [1, 0]  # ratio or num
    ratio_list = [0.1, 0.01]  # [train_ratio,val_ratio]

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])
    evaluator = Evaluator(num_class=class_count)

    for exp_idx,curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)
        single_experiment_name = 'run{}_seed{}'.format(str(exp_idx), str(curr_seed))
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis')
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
            print("makedirs {}".format(save_vis_folder))

        save_weight_path = os.path.join(save_single_experiment_folder, "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder, 'result_tr{}_val{}.txt'.format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder, 'pred_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder, 'gt_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list,
                                                                                            num_list,
                                                                                            gt_reshape,
                                                                                            class_count,
                                                                                            flag_list[1])
        # save = True
        # if save == False:
        #     save_dir = "./data_split_indices"
        #     print("Sampling.....")
        #     os.makedirs(save_dir, exist_ok=True)

        #     train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list,
        #                                                                                                 num_list,
        #                                                                                                 gt_reshape,
        #                                                                                                 class_count,
        #                                                                                                 flag_list[1])
        #     np.save(os.path.join(save_dir, "train_data_index_LN1.npy"), train_data_index)
        #     np.save(os.path.join(save_dir, "val_data_index_LN1.npy"),   val_data_index)
        #     np.save(os.path.join(save_dir, "test_data_index_LN1.npy"),  test_data_index)
        #     np.save(os.path.join(save_dir, "all_data_index_LN1.npy"),   all_data_index)

        #     print("✅ Dataset split indices saved.")
        # else:
        #     load_dir = "./data_split_indices"

        #     train_data_index = np.load(os.path.join(load_dir, "train_data_index.npy"))
        #     val_data_index   = np.load(os.path.join(load_dir, "val_data_index.npy"))
        #     test_data_index  = np.load(os.path.join(load_dir, "test_data_index.npy"))
        #     all_data_index   = np.load(os.path.join(load_dir, "all_data_index.npy"))

        #     print("✅ Dataset split indices loaded.")
        index = (train_data_index, val_data_index, test_data_index)
        train_label, val_label, test_label = data_load_operate.generate_image_iter(data, height, width, gt_reshape, index)
        new_label100, _ = split_label_map_by_spectra(data, train_label.numpy(), class_count, n_subclasses=50)    
        new_label50, _ = split_label_map_by_spectra(data, train_label.numpy(), class_count, n_subclasses=30)     
        new_label30, _ = split_label_map_by_spectra(data, train_label.numpy(), class_count, n_subclasses=20)     
        newcluter100, _  = seeded_kmeans_hsi(data, new_label100)  #cluster map1
        newcluter50, _ = seeded_kmeans_hsi(data, new_label50)     #cluster map2
        newcluter30, _  = seeded_kmeans_hsi(data, new_label30)    #cluster map1
        num_label100 = np.unique(newcluter100, return_counts=True)[1]   # number of pixels per cluster
        num_label50 = np.unique(newcluter50, return_counts=True)[1]
        num_label30 = np.unique(newcluter30, return_counts=True)[1]
        new_label100 = torch.from_numpy(newcluter100).long()
        new_label50 = torch.from_numpy(newcluter50).long()
        new_label30 = torch.from_numpy(newcluter30).long()
        cluster_map(newcluter30, newcluter50, newcluter100)
        #cluster_map, _, _ = supervised_subkmeans_from_label(data, train_label.numpy(), n_classes=num_clusters, sub_clusters_per_class=3, random_state=42)

        #per_cluster_num = np.unique(cluster_map, return_counts=True)[1]

        #print(f"\nClustering completed!")
        #print(f"Cluster map shape: {cluster_map.shape}")
        #print(f"Unique clusters: {np.unique(cluster_map)}")

        #sorted_indices, sorted_counts = visualize_clusters(data, cluster_map, n_clusters=num_clusters*3)

        #sorted_indices, head_clusters, tail_clusters = analyze_cluster_statistics(cluster_map)

        # Return the cluster map for further use
        #print(f"\nCluster map is ready for use!")
        #print(f"Shape: {cluster_map.shape}")
        #print(f"Value range: {cluster_map.min()} to {cluster_map.max()}")

        # build Model
        # net = mHC_MambaHSI_Integrated(in_channels=200,
        #                                 hidden_dim=400,
        #                                 num_classes=16,
        #                                 token_num=10)
        indian_pines_wl = np.linspace(400, 2500, 200)
        net = ImageHyperConnectionTransformer(
                    image_size=(145, 145),
                    patch_size=(1, 1),
                    in_channels=200,
                    num_classes=16,
                    dim=320,
                    n_layers=6,
                    n_heads=4,
                    rate=None,
                    dropout=0.1,
                    pool_size=2,
                    mask_ratio=0.0,
                    wavelengths=indian_pines_wl
                ).to(device)
        vis_idx   = np.where((indian_pines_wl >= 400) & (indian_pines_wl < 700))[0]
        nir_idx   = np.where((indian_pines_wl >= 700) & (indian_pines_wl < 1000))[0]
        swir1_idx = np.where((indian_pines_wl >= 1000) & (indian_pines_wl < 1800))[0]
        swir2_idx = np.where((indian_pines_wl >= 1800) & (indian_pines_wl <= 2500))[0]
        band_groups = {
            "VIS":   vis_idx,
            "NIR":   nir_idx,
            "SWIR1": swir1_idx,
            "SWIR2": swir2_idx,
        }
        for name, idx in band_groups.items():
            x_g = img[:, :, idx]       
            x_g = x_g.mean(axis=-1)
            fig = plt.figure()
            plt.imshow(x_g)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(name + '_LN.png', bbox_inches="tight")
            plt.close(fig)
        # net = ImageHyperConnectionTransformer(
        #     image_size=(145, 145),
        #     patch_size=(1, 1),
        #     in_channels=200,
        #     num_classes=16,
        #     dim=256,
        #     n_layers=6,
        #     n_heads=4,
        #     rate=2,
        #     dropout=0.1,
        #     pool_size=2,
        #     mask_ratio=0.0,
        # ).to(device) 
        #net = MambaHSI(in_channels=200,hidden_dim=512,num_classes=16)
        #net = MambaHSI(in_channels=channels, num_classes=class_count, hidden_dim=32, num_clusters=num_clusters*3, sparsity_ratio=1.0)
        #net.load_state_dict(torch.load('/mnt/storage/Hack14/MambaHSI/RUNS/MambaHSI/IN/run0_seed0/best_tr10_val10.pth'))
        #.eval()
        logger.info(paras_dict)
        logger.info(net)

        x = transform(np.array(img))
        x = x.unsqueeze(0).float().to(device)

        train_label = train_label.to(device)
        test_label = test_label.to(device)
        val_label = val_label.to(device)

        # ############################################
        # val_label = test_label
        # ############################################

        net.to(device)
        #centers_cluster = net.mamba1.spa_mamba.mamba.cluster_head.cluster_centers.detach().cpu().numpy()
        #visualize_cluster_centers(centers_cluster, 0)
        train_loss_list = [100]
        train_acc_list = [0]
        val_loss_list = [100]
        val_acc_list = [0]

        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

        logger.info(optimizer)
        best_loss = 99999
        # if record_computecost:
            # net.eval()
            # flops, macs1, para = calculate_flops(model=net,
            #                                      input_shape=(1, x.shape[1], x.shape[2], x.shape[3]), )
            # logger.info("para:{}\n,flops:{}".format(para, flops))

        tic1 = time.perf_counter()
        best_val_acc = 0
        margin = 5
        h_mid = 300 // 2
        w_mid = 300 // 2

        for epoch in range(max_epoch):
            y_train = train_label.unsqueeze(0)
            train_acc_sum, trained_samples_counter = 0.0, 0
            batch_counter, train_loss_sum = 0, 0
            time_epoch = time.time()
            loss_dict = {}

            net.train()
            if split_image:
                # Top-left
                x_part1 = x[:, :, :h_mid + margin, :w_mid + margin]
                y_part1 = y_train[:, :h_mid + margin, :w_mid + margin]

                # Top-right
                x_part2 = x[:, :, :h_mid + margin, w_mid - margin:]
                y_part2 = y_train[:, :h_mid + margin, w_mid - margin:]

                # Bottom-left
                x_part3 = x[:, :, h_mid - margin:, :w_mid + margin]
                y_part3 = y_train[:, h_mid - margin:, :w_mid + margin]

                # Bottom-right
                x_part4 = x[:, :, h_mid - margin:, w_mid - margin:]
                y_part4 = y_train[:, h_mid - margin:, w_mid - margin:]
                y_pred_part1 = net(x_part1)

                ls1 = head_loss(loss_func,y_pred_part1, y_part1.long())
                optimizer.zero_grad()
                ls1.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                y_pred_part2 = net(x_part2)
                ls2 = head_loss(loss_func,y_pred_part2, y_part2.long())
                optimizer.zero_grad()
                ls2.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                
                y_pred_part3 = net(x_part3)
                ls3 = head_loss(loss_func,y_pred_part3, y_part3.long())
                optimizer.zero_grad()
                ls3.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                
                y_pred_part4 = net(x_part4)
                ls4 = head_loss(loss_func,y_pred_part4, y_part4.long())
                optimizer.zero_grad()
                ls4.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                #logger.info('Iter:{}|loss:{}'.format(epoch, (ls1 + ls2).detach().cpu().numpy()))

            else:
                #try:
                #y_pred, loss2 = net(x, [num_label100, num_label50, num_label30], new_label100.to(device), new_label50.to(device), new_label30.to(device))
                y_pred = net(x, gt_reshape, epoch)
                ls = head_loss(loss_func,y_pred, y_train.long())
                #cluster_labels_list = [new_label100.to(device), new_label50.to(device), new_label30.to(device)]
                optimizer.zero_grad()
                ls.backward()
                optimizer.step()
                #print(net.mamba1.spa_mamba.mamba.cluster_head.cluster_centers)
                #logger.info('Iter:{}|loss:{}|'.format(epoch, ls.detach().cpu().numpy()))
                # except:
                #     optimizer.zero_grad()
                #     torch.cuda.empty_cache()
                #     split_image=True
                #     x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                #     y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                #     x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                #     y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]
                #
                #     y_pred_part1, loss2 = net(x_part1, [num_label100, num_label50, num_label30], new_label100.to(device), new_label50.to(device), new_label30.to(device))
                #     ls1 = head_loss(loss_func, y_pred_part1, y_part1.long())+ loss2
                #     optimizer.zero_grad()
                #     ls1.backward()
                #     optimizer.step()
                #
                #     y_pred_part2 = net(x_part2)
                #     ls2 = head_loss(loss_func, y_pred_part2, y_part2.long())
                #     optimizer.zero_grad()
                #     ls2.backward()
                #     optimizer.step()
                #
                #     logger.info(
                #         'Iter:{}|loss:{}'.format(epoch, (ls1 + ls2).detach().cpu().numpy()))

            #torch.cuda.empty_cache()
            # evaluate stage
            net.eval()
            if (epoch+1) % 10 == 0:
                with torch.no_grad():
                    if split_image is True:                                
                        B, C, H, W = x.shape
                        margin = 5
                        h_mid = H // 2
                        w_mid = W // 2

                        # ================================
                        # 1. Split into 4 overlapping parts
                        # ================================
                        x_part1 = x[:, :, :h_mid + margin, :w_mid + margin]          # TL
                        x_part2 = x[:, :, :h_mid + margin, w_mid - margin:]          # TR
                        x_part3 = x[:, :, h_mid - margin:, :w_mid + margin]          # BL
                        x_part4 = x[:, :, h_mid - margin:, w_mid - margin:]          # BR

                        # ================================
                        # 2. Forward pass (logits)
                        # ================================
                        logits1 = net(x_part1)   # (B, C, h1, w1)
                        logits2 = net(x_part2)
                        logits3 = net(x_part3)
                        logits4 = net(x_part4)

                        num_classes = logits1.shape[1]

                        # ================================
                        # 3. Init full canvas
                        # ================================
                        full_logits = torch.zeros((B, num_classes, H, W),
                                                device=x.device)
                        count_map = torch.zeros((B, 1, H, W),
                                                device=x.device)

                        # ================================
                        # 4. Stitch logits back
                        # ================================
                        # Top-left
                        full_logits[:, :, :h_mid + margin, :w_mid + margin] += logits1
                        count_map[:, :, :h_mid + margin, :w_mid + margin] += 1

                        # Top-right
                        full_logits[:, :, :h_mid + margin, w_mid - margin:] += logits2
                        count_map[:, :, :h_mid + margin, w_mid - margin:] += 1

                        # Bottom-left
                        full_logits[:, :, h_mid - margin:, :w_mid + margin] += logits3
                        count_map[:, :, h_mid - margin:, :w_mid + margin] += 1

                        # Bottom-right
                        full_logits[:, :, h_mid - margin:, w_mid - margin:] += logits4
                        count_map[:, :, h_mid - margin:, w_mid - margin:] += 1

                        # ================================
                        # 5. Fuse overlaps (average)
                        # ================================
                        full_logits = full_logits / count_map
                        pred_full = torch.argmax(full_logits, dim=1).cpu().numpy()  # (B, H, W)

                        # ================================
                        # 6. Accuracy with GT
                        # ================================
                        if val_label.ndim == 2:
                            y_val = val_label.unsqueeze(0)
                        else:
                            y_val = val_label
                        Y_val_np = val_label.cpu().numpy()
                        Y_val_255 = np.where(Y_val_np==-1,255,Y_val_np)
                        evaluator.add_batch(np.expand_dims(Y_val_255,axis=0), pred_full)
                        OA = evaluator.Pixel_Accuracy()
                        mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                        mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                        Kappa = evaluator.Kappa()
                        logger.info('Evaluate {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA,mAcc,Kappa,mIOU,IOU,Acc))
                        #if (epoch+1) % 50 == 0:
                        #    centers_cluster = net.mamba1.spa_mamba.mamba.cluster_head.cluster_centers.detach().cpu().numpy()
                        #    visualize_cluster_centers(centers_cluster, epoch)
                        # save weight
                        if OA>=best_val_acc:
                            best_epoch = epoch + 1
                            best_val_acc = OA
                            # torch.save(net,save_weight_path)
                            torch.save(net.state_dict(), save_weight_path)
                            print("saved weights===========")
                            # save_epoch_weight_path = os.path.join(save_folder,'{}.pth'.format(str(epoch+1)))
                            # torch.save(net.state_dict(), save_epoch_weight_path)
                        if (epoch+1)%50==0:
                            save_single_predict_path = os.path.join(save_vis_folder,'predict_{}.png'.format(str(epoch+1)))
                            save_single_gt_path = os.path.join(save_vis_folder,'gt.png')
                            vis_a_image(gt,pred_full,save_single_predict_path, save_single_gt_path)
                    else:
                        seg_logits = net(x)
                        y_val = val_label.unsqueeze(0)
                        # seg_logits = resize(input=output_val,
                        #                     size=y_val.shape[1:],
                        #                     mode='bilinear',
                        #                     align_corners=True)
                        predict = torch.argmax(seg_logits,dim=1).cpu().numpy()
                        Y_val_np = val_label.cpu().numpy()
                        Y_val_255 = np.where(Y_val_np==-1,255,Y_val_np)
                        evaluator.add_batch(np.expand_dims(Y_val_255,axis=0),predict)
                        OA = evaluator.Pixel_Accuracy()
                        mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                        mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                        Kappa = evaluator.Kappa()
                        logger.info('Evaluate {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA,mAcc,Kappa,mIOU,IOU,Acc))
                        #if (epoch+1) % 50 == 0:
                        #    centers_cluster = net.mamba1.spa_mamba.mamba.cluster_head.cluster_centers.detach().cpu().numpy()
                        #    visualize_cluster_centers(centers_cluster, epoch)
                        # save weight
                        if OA>=best_val_acc:
                            best_epoch = epoch + 1
                            best_val_acc = OA
                            # torch.save(net,save_weight_path)
                            torch.save(net.state_dict(), save_weight_path)
                            print("saved weights===========")
                            # save_epoch_weight_path = os.path.join(save_folder,'{}.pth'.format(str(epoch+1)))
                            # torch.save(net.state_dict(), save_epoch_weight_path)
                        if (epoch+1)%50==0:
                            save_single_predict_path = os.path.join(save_vis_folder,'predict_{}.png'.format(str(epoch+1)))
                            save_single_gt_path = os.path.join(save_vis_folder,'gt.png')
                            vis_a_image(gt,predict,save_single_predict_path, save_single_gt_path)

                # net.train()
            torch.cuda.empty_cache()


        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        pred_test = []

        load_weight_path = save_weight_path
        #net.update_params = None
        # best_net = copy.deepcopy(net)
        # best_net = mHC_MambaHSI_Integrated(in_channels=200,
        #                                 hidden_dim=400,
        #                                 num_classes=16,
        #                                 token_num=10)

        # best_net = ImageHyperConnectionTransformer(
        #             image_size=(145, 145),
        #             patch_size=(1, 1),
        #             in_channels=200,
        #             num_classes=16,
        #             dim=256,
        #             n_layers=6,
        #             n_heads=4,
        #             rate=4,
        #             dropout=0.1,
        #             pool_size=2,
        #             mask_ratio=0.0,
        #         ).to(device) 
        #best_net = MambaHSI(in_channels=200,hidden_dim=512,num_classes=16)
        #best_net = MambaHSI(in_channels=channels, num_classes=class_count, hidden_dim=32, num_clusters=num_clusters*3)
        net.load_state_dict(torch.load(load_weight_path))
        net.eval()
        test_evaluator = Evaluator(num_class=class_count)
        with torch.no_grad():
            if split_image is True:
                test_evaluator.reset()
                
                B, C, H, W = x.shape
                margin = 5
                h_mid = H // 2
                w_mid = W // 2

                # ================================
                # 1. Split into 4 overlapping parts
                # ================================
                x_part1 = x[:, :, :h_mid + margin, :w_mid + margin]          # TL
                x_part2 = x[:, :, :h_mid + margin, w_mid - margin:]          # TR
                x_part3 = x[:, :, h_mid - margin:, :w_mid + margin]          # BL
                x_part4 = x[:, :, h_mid - margin:, w_mid - margin:]          # BR

                # ================================
                # 2. Forward pass (logits)
                # ================================
                logits1 = net(x_part1)   # (B, C, h1, w1)
                logits2 = net(x_part2)
                logits3 = net(x_part3)
                logits4 = net(x_part4)

                num_classes = logits1.shape[1]

                # ================================
                # 3. Init full canvas
                # ================================
                full_logits = torch.zeros((B, num_classes, H, W),
                                        device=x.device)
                count_map = torch.zeros((B, 1, H, W),
                                        device=x.device)

                # ================================
                # 4. Stitch logits back
                # ================================
                # Top-left
                full_logits[:, :, :h_mid + margin, :w_mid + margin] += logits1
                count_map[:, :, :h_mid + margin, :w_mid + margin] += 1

                # Top-right
                full_logits[:, :, :h_mid + margin, w_mid - margin:] += logits2
                count_map[:, :, :h_mid + margin, w_mid - margin:] += 1

                # Bottom-left
                full_logits[:, :, h_mid - margin:, :w_mid + margin] += logits3
                count_map[:, :, h_mid - margin:, :w_mid + margin] += 1

                # Bottom-right
                full_logits[:, :, h_mid - margin:, w_mid - margin:] += logits4
                count_map[:, :, h_mid - margin:, w_mid - margin:] += 1

                # ================================
                # 5. Fuse overlaps (average)
                # ================================
                full_logits = full_logits / count_map
                pred_full = torch.argmax(full_logits, dim=1).cpu().numpy()  # (B, H, W)
                
                # ================================
                # 6. Accuracy with GT
                # ================================
                if test_label.ndim == 2:
                    y_val = test_label.unsqueeze(0)
                else:
                    y_val = test_label
                Y_val_np = test_label.cpu().numpy()
                Y_val_255 = np.where(Y_val_np==-1,255,Y_val_np)
                            
                #output_test = net(x, epoch=999)
                #output_test = best_net(x, [num_label100, num_label50, num_label30], new_label100.to(device), new_label50.to(device), new_label30.to(device))

                #y_test = test_label.unsqueeze(0)
                #seg_logits_test = output_test
                # seg_logits_test = resize(input=seg_logits_test,
                #                     size=y_test.shape[1:],
                #                     mode='bilinear',
                #                     align_corners=True)
                #predict_test = torch.argmax(seg_logits_test, dim=1).cpu().numpy()
                Y_test_np = test_label.cpu().numpy()
                Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)
                test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), pred_full)
                OA_test = test_evaluator.Pixel_Accuracy()
                mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
                mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
                Kappa_test = test_evaluator.Kappa()
                logger.info('|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(OA_test, mAcc_test, Kappa_test, mIOU_test, IOU_test,
                                                                                        Acc_test))
                vis_a_image(gt, pred_full, predict_save_path, gt_save_path)
            else:
                test_evaluator.reset()
                output_test = net(x, gt, epoch=999)

                seg_logits_test = test_label.unsqueeze(0)
                # seg_logits_test = resize(input=output_test,
                #                     size=y_test.shape[1:],
                #                     mode='bilinear',
                #                     align_corners=True)
                predict_test = torch.argmax(output_test, dim=1).cpu().numpy()
                Y_test_np = test_label.cpu().numpy()
                Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)
                test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), predict_test)
                OA_test = test_evaluator.Pixel_Accuracy()
                mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
                mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
                Kappa_test = evaluator.Kappa()
                logger.info('Test {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA_test, mAcc_test, Kappa_test, mIOU_test, IOU_test,
                                                                                        Acc_test))
                vis_a_image(gt, predict_test, predict_save_path, gt_save_path)
        # Output infors
        f = open(results_save_path, 'a+')
        str_results = '\n======================' \
                      + " exp_idx=" + str(exp_idx) \
                      + " seed=" + str(curr_seed) \
                      + " learning rate=" + str(learning_rate) \
                      + " epochs=" + str(max_epoch) \
                      + " train ratio=" + str(ratio_list[0]) \
                      + " val ratio=" + str(ratio_list[1]) \
                      + " ======================" \
                      + "\nOA=" + str(OA_test) \
                      + "\nAA=" + str(mAcc_test) \
                      + '\nkpp=' + str(Kappa_test) \
                      + '\nmIOU_test:' + str(mIOU_test) \
                      + "\nIOU_test:" + str(IOU_test) \
                      + "\nAcc_test:" + str(Acc_test) + "\n"
        logger.info(str_results)
        f.write(str_results)
        f.close()

        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)

        torch.cuda.empty_cache()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    logger.info('List of OA:', list(OA_ALL))
    logger.info('List of AA:', list(AA_ALL))
    logger.info('List of KPP:', list(KPP_ALL))
    logger.info('OA=', round(np.mean(OA_ALL) * 100, 2), '+-', round(np.std(OA_ALL) * 100, 2))
    logger.info('AA=', round(np.mean(AA_ALL) * 100, 2), '+-', round(np.std(AA_ALL) * 100, 2))
    logger.info('Kpp=', round(np.mean(KPP_ALL) * 100, 2), '+-', round(np.std(KPP_ALL) * 100, 2))
    logger.info('Acc per class=', np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2), '+-',
          np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2))

    logger.info("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    logger.info("Average testing time=", round(np.mean(Test_Time_ALL) * 1000, 2), '+-',
          round(np.std(Test_Time_ALL) * 1000, 3))

    # Output infors
    mean_result_path = os.path.join(save_folder,'mean_result.txt')
    f = open(mean_result_path, 'w')
    str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + 'times runs ********************' \
                  + '\nList of OA:' + str(list(OA_ALL)) \
                  + '\nList of AA:' + str(list(AA_ALL)) \
                  + '\nList of KPP:' + str(list(KPP_ALL)) \
                  + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                  + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                  + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(
        round(np.std(KPP_ALL) * 100, 2)) \
                  + '\nAcc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' + str(
        np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)) \
                  + "\nAverage training time=" + str(
        np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(
        np.round(np.std(Train_Time_ALL), decimals=3)) \
                  + "\nAverage testing time=" + str(
        np.round(np.mean(Test_Time_ALL) * 1000, decimals=2)) + '+-' + str(
        np.round(np.std(Test_Time_ALL) * 100, decimals=3))
    f.write(str_results)
    f.close()

    del net
