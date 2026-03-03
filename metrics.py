import os
import glob

import numpy as np
import torch
import torchvision.transforms as tvf

from PIL import Image
from scipy import linalg
from typing import Optional, Union, Tuple
from argparse import ArgumentParser
from tqdm import tqdm
from dataclasses import dataclass

from inception import InceptionV3

def _get_default_workers() -> int:
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()
    return min(num_cpus, 8) if num_cpus is not None else 0


@dataclass
class ImgGenEvalDefaultConfigs:
    batch_size: int = 50
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = _get_default_workers()


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        img = tvf.ToTensor()(img)
        return img


class ImageArrayDataset(torch.utils.data.Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)  # self.array.shape[0]

    def __getitem__(self, i):
        img = self.array[i]
        img = tvf.ToTensor()(img)
        return img


def get_activations(
    files_or_array: Union[list, np.ndarray],
    model: InceptionV3,
    batch_size: int = ImgGenEvalDefaultConfigs.batch_size,
    device: torch.device = ImgGenEvalDefaultConfigs.device,
    num_workers: int = ImgGenEvalDefaultConfigs.num_workers,
):
    """Calculates the InceptionV3 activations about sFID & FID for all images."""
    # Reference: https://github.com/mseitzer/pytorch-fid/blob/b9c18118d082cbd263c1b8963fc4221dc1cbb659/src/pytorch_fid/fid_score.py#L113
    model.eval()

    if isinstance(files_or_array, np.ndarray):
        dataset = ImageArrayDataset(files_or_array)
    elif isinstance(files_or_array, list) and isinstance(files_or_array[0], str):
        dataset = ImagePathDataset(files_or_array)
    else:
        raise ValueError("files_or_array must be a list of image file paths or a numpy array of images.")

    n_images = len(dataset)
    batch_size = batch_size if batch_size <= n_images else n_images
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_fid_feats = []
    pred_sfid_feats = []
    pred_is_logits = []

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            # pred = model(batch)[0]
            pred_raw = model(batch)
            pred_fid, pred_sfid, pred_is = pred_raw  # [bsz, 2048, 1, 1], [bsz, 192, 17, 17], [bsz, 1008]
            
            bsz       = pred_fid.shape[0]
            pred_fid  = pred_fid.reshape(bsz, -1)   # (bsz, 2048)
            pred_sfid = pred_sfid[:, :7, :, :].permute(0, 2, 3, 1)
            pred_sfid = pred_sfid.reshape(bsz, -1)  # (bsz, 2023)

        pred_fid_feats.append(pred_fid)    # 2048
        pred_sfid_feats.append(pred_sfid)  # 2023
        pred_is_logits.append(pred_is)     # 1008

    pred_arr_fid  = torch.cat(pred_fid_feats, dim=0)
    pred_arr_sfid = torch.cat(pred_sfid_feats, dim=0)
    pred_arr_is   = torch.cat(pred_is_logits, dim=0)

    return pred_arr_fid, pred_arr_sfid, pred_arr_is


def calculate_frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ):
    """
    Compute the Frechet distance between two sets of statistics.
    """
    # https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py#L72-L115
    # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_inception_score(
        act_is: torch.Tensor,
        split_size: int = 5000,
    ) -> float:
    """Calculates the Inception Score (PyTorch version)"""
    # https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/evaluations/evaluator.py#L179-L192
    scores = []
    for i in range(0, act_is.shape[0], split_size):
        part = act_is[i : i + split_size]
        # p(y) = E_x[p(y|x)]
        p_y = torch.mean(part, dim=0, keepdim=True)
        # D_KL(p(y|x) || p(y)) = p(y|x) * (log(p(y|x)) - log(p(y)))
        kl = part * (torch.log(part) - torch.log(p_y))
        kl = torch.mean(torch.sum(kl, dim=1))
        # undo log -> exp
        scores.append(torch.exp(kl))
    inception_score_tensor = torch.mean(torch.stack(scores))
    return inception_score_tensor.item()


def manifold_radii(
    features: torch.Tensor,
    nhood_sizes: Tuple[int] = (3, ),
    clamp_to_percentile: Optional[float] = None,
) -> torch.Tensor:
    """ Compute manifold radii for each feature vector. (PyTorch version) """
    # https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/evaluations/evaluator.py#L248
    nhood_indices = torch.tensor(
        nhood_sizes,
        device=features.device,
        dtype=torch.long
    )
    
    # 0, 1, ..., max(nhood_sizes)
    k_max = max(nhood_sizes) + 1

    pairwise_dists = torch.cdist(features, features, p=2.0)  # (N, N), originally need .pow(2)

    # largest=False -> k-nearest neighbours
    # dim=1: every row, operate independently
    kth_values, _ = torch.topk(
        pairwise_dists, 
        k=k_max, 
        dim=1, 
        largest=False
    )  # (N, k_max)

    radii = kth_values.index_select(dim=1, index=nhood_indices)  # (N, len(nhood_sizes))

    if clamp_to_percentile is not None:
        # (0-100) -> (0-1)
        q = clamp_to_percentile / 100.0
        # dim = 0, batch dimension
        max_distances = torch.quantile(radii, q, dim=0)
        # clip radii
        radii = torch.where(radii > max_distances, 0.0, radii)

    return radii


def calculate_precision_recall(
        act_src: torch.Tensor,
        act_ref: torch.Tensor,
    ):
    """ Evaluate precision and recall efficiently (PyTorch version) """
    # https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/evaluations/evaluator.py#L325
    radii_ref = manifold_radii(act_ref)
    radii_src = manifold_radii(act_src)
    # print(torch.cuda.memory_allocated(device) / 1024**3)

    dist_matrix = torch.cdist(act_ref, act_src, p=2.0)  # (10000, 50000), originally need .pow(2)
    dist_matrix_exp = dist_matrix.unsqueeze(-1)  # (10000, 50000, 1)

    act_ref_status = torch.any(dist_matrix_exp <= radii_src, dim=1).squeeze(-1)
    act_src_status = torch.any(dist_matrix_exp <= radii_ref.unsqueeze(-1), dim=0).squeeze(-1)

    precision = act_src_status.to(torch.float64).mean().item()
    recall = act_ref_status.to(torch.float64).mean().item()
    return precision, recall


def inception_forward_or_load(
        path: str,
        model: InceptionV3,
        batch_size: int = ImgGenEvalDefaultConfigs.batch_size,
        device: torch.device = ImgGenEvalDefaultConfigs.device,
        num_workers: int = ImgGenEvalDefaultConfigs.num_workers,
    ):
    if path.endswith(".npz"):
        required_keys = {"mu", "sigma", "mu_s", "sigma_s", "arr_0"}
        with np.load(path) as f:
            if not required_keys.issubset(f.keys()):
                raise ValueError(f"NPZ file {path} is missing required keys: {required_keys - set(f.keys())}")
            # for FID & sFID
            mu, sigma     = f["mu"], f["sigma"]
            mu_s, sigma_s = f["mu_s"], f["sigma_s"]
            array         = f["arr_0"]
            score         = f.get("score", None)
        
        if not isinstance(array, np.ndarray):
            raise ValueError(f"Loaded arr_0 from npz must be a numpy array, but got {type(array)}")
        elif array.ndim == 4:
            # (N, 256, 256, 3)
            act, act_s, act_is = get_activations(array, model, batch_size, device, num_workers)
            incep_score = calculate_inception_score(act_is)
        elif array.ndim == 2:
            # (N, 2048)
            act = torch.from_numpy(array).to(device)
            if score is None:
                raise ValueError("If given fid activation in npz file, inception score must be given as well")
            incep_score = score.item()
        else:
            raise ValueError(f"Loaded arr_0 from npz must be either (N, 256, 256, 3) or (N, 2048), but got shape {array.shape}")
    else:
        image_files = glob.glob(os.path.join(path, "*"))
        image_file_list = [
            file for file in image_files
            if file.split(".")[-1].lower() in {
                "bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"
            }
        ]

        act, act_s, act_is = get_activations(image_file_list, model, batch_size, device, num_workers)

        # for FID & sFID
        mu      = torch.mean(act, dim=0).cpu().numpy()     # (N, 2048) -> (2048, )
        sigma   = act.t().cov().cpu().numpy()     # (N, 2048) -> (2048, N) -> (2048, 2048)
        mu_s    = torch.mean(act_s, dim=0).cpu().numpy()   # (N, 2023) -> (2023, )
        sigma_s = act_s.t().cov().cpu().numpy()

        incep_score = calculate_inception_score(act_is)

    return act, mu, sigma, mu_s, sigma_s, incep_score


def basic_path_check(path: str, prefix: str):
    # ensure src_image_folder, ref_image_folder in [ None (0) | npz file (1, exists) | image folder (2, exists) ]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{prefix} path {path} does not exist.")
    
    is_dir = os.path.isdir(path)
    is_npz = path.endswith(".npz")
    if not is_dir and not is_npz:
        raise ValueError(f"{prefix} path must be a directory or an npz file, but got {path}.")
    return is_npz


def save_npz_path_check(path: Optional[str], prefix: str):
    # ensure src_npz_path, ref_npz_path in [ None (0) | npz file (1, to save, not exists yet) ]
    if path is None:
        return

    if os.path.exists(path):
        raise ValueError(f"{prefix} path {path} is already existing.")
    
    if not path.endswith(".npz"):
        raise ValueError(f"{prefix} path must be an npz file (to save, not exists yet), but got {path}.")


def save_to_npz(path: Optional[str], mu: np.ndarray, sigma: np.ndarray, mu_s: np.ndarray, sigma_s: np.ndarray, array: torch.Tensor, score: float):
    if path is None: return

    print("Saving npz...")
    np.savez_compressed(path, mu=mu, sigma=sigma, mu_s=mu_s, sigma_s=sigma_s, arr_0=array.cpu().numpy(), score=score)
    print(f"Saved npz to {path}.")


def compute_metrics(
        src_path: str,
        ref_path: Optional[str] = None,
        save_src_npz_path: Optional[str] = None,
        save_ref_npz_path: Optional[str] = None,
        inception_weights: Optional[str] = None,
        batch_size: int = ImgGenEvalDefaultConfigs.batch_size,
        device: torch.device = ImgGenEvalDefaultConfigs.device,
        num_workers: int = ImgGenEvalDefaultConfigs.num_workers,
    ):
    src_is_npz = basic_path_check(src_path, prefix="source")
    ref_is_npz = basic_path_check(ref_path, prefix="reference") if ref_path is not None else True
    save_npz_path_check(save_src_npz_path, prefix="source npz save")
    save_npz_path_check(save_ref_npz_path, prefix="reference npz save")

    model = InceptionV3(inception_weights=inception_weights).to(device)
    
    act_src, mu_src, sigma_src, mu_s_src, sigma_s_src, score_src = inception_forward_or_load(
        src_path, model, batch_size, device, num_workers
    )

    save_to_npz(save_src_npz_path, mu_src, sigma_src, mu_s_src, sigma_s_src, act_src, score_src)

    if ref_path is not None:
        act_ref, mu_ref, sigma_ref, mu_s_ref, sigma_s_ref, score_ref = inception_forward_or_load(
            ref_path, model, batch_size, device, num_workers
        )

        save_to_npz(save_ref_npz_path, mu_ref, sigma_ref, mu_s_ref, sigma_s_ref, act_ref, score_ref)

        fid_value = calculate_frechet_distance(mu_src, sigma_src, mu_ref, sigma_ref)
        sfid_value = calculate_frechet_distance(mu_s_src, sigma_s_src, mu_s_ref, sigma_s_ref)
        precision, recall = calculate_precision_recall(act_src, act_ref)

        print("FID: {:.2f}".format(fid_value))
        print("sFID: {:.2f}".format(sfid_value))
        print("Inception Score (src): {:.2f}".format(score_src))
        print("Inception Score (ref): {:.2f}".format(score_ref))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))

    else:
        print("Inception Score: {:.2f}".format(score_src))
    
    del model

def main():
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=ImgGenEvalDefaultConfigs.batch_size, help="batch size to use")
    parser.add_argument("--num-workers", type=int, default=ImgGenEvalDefaultConfigs.num_workers, help="number of dataloader workers to use")
    parser.add_argument("--device", type=str, default=None, help="device to use, such as cuda, cuda:0 or cpu")
    parser.add_argument("--weights", type=str, default=None, help="path to inception weights file")
    parser.add_argument("--save-src-npz-path", type=str, default=None, help="path to save an npz file from a directory of source samples.")
    parser.add_argument("--save-ref-npz-path", type=str, default=None, help="path to save an npz file from a directory of reference samples.")
    parser.add_argument("--src-path", type=str, required=True, help="path to the directory of source samples or npz file to evaluate")
    parser.add_argument("--ref-path", type=str, default=None, help="path to the directory of reference samples or npz file to evaluate")
    args = parser.parse_args()

    batch_size = args.batch_size
    num_workers = args.num_workers
    device = ImgGenEvalDefaultConfigs.device if args.device is None else torch.device(args.device)

    compute_metrics(
        src_path=args.src_path,
        ref_path=args.ref_path,
        save_src_npz_path=args.save_src_npz_path,
        save_ref_npz_path=args.save_ref_npz_path,
        inception_weights=args.weights,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )

if __name__ == "__main__":
    main()
