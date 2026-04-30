import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import get_args_parser
from models import build_model
from utils.dataset import (
    Dataset_Creator,
    Dataset_Creator_Chameleon,
    Dataset_Creator_Chameleon_SD,
    Dataset_Creator_GenImage,
)


HYPOTHESIS_NAMES = ["base", "artifact", "structure", "patch"]


def _as_float(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return float(value)


def _safe_metrics(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = y_score > 0.5
    out = {
        "acc": _as_float(accuracy_score(y_true, y_pred)),
        "real_acc": 0.0,
        "fake_acc": 0.0,
        "ap": 0.0,
    }
    if np.any(y_true == 0):
        out["real_acc"] = _as_float(accuracy_score(y_true[y_true == 0], y_pred[y_true == 0]))
    if np.any(y_true == 1):
        out["fake_acc"] = _as_float(accuracy_score(y_true[y_true == 1], y_pred[y_true == 1]))
    if len(np.unique(y_true)) > 1:
        out["ap"] = _as_float(average_precision_score(y_true, y_score))
    return out


def _dataset_creator(args):
    common = dict(
        dataset_path=args.dataset_path,
        batch_size=args.evalbatchsize,
        num_workers=args.num_workers,
        img_resolution=args.img_resolution,
        crop_resolution=args.crop_resolution,
    )
    if args.dataset == "UniversalFakeDetect":
        return Dataset_Creator(**common)
    if args.dataset == "GenImage":
        return Dataset_Creator_GenImage(**common)
    if args.dataset == "Chameleon":
        return Dataset_Creator_Chameleon(**common)
    if args.dataset == "Chameleon_SD":
        return Dataset_Creator_Chameleon_SD(**common)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        return {"missing": len(missing), "unexpected": len(unexpected), "shape_skipped": 0}
    except RuntimeError:
        own_state = model.state_dict()
        filtered = {}
        skipped = 0
        for key, value in state_dict.items():
            clean_key = key[7:] if key.startswith("module.") else key
            if clean_key in own_state and own_state[clean_key].shape == value.shape:
                filtered[clean_key] = value
            else:
                skipped += 1
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        return {"missing": len(missing), "unexpected": len(unexpected), "shape_skipped": skipped}


def _empty_accumulator():
    return {
        "labels": [],
        "logits": {name: [] for name in ["final", "base", "artifact", "structure", "patch", "uniform"]},
        "gate_probs": [],
        "all_logits": [],
        "base_family": {name: [] for name in ["orig", "artifact", "structure", "patch"]},
        "tile_entropy": [],
        "tile_variance": [],
    }


def _append_batch(acc, labels, outputs):
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    all_logits = outputs["all_logits"].detach().cpu()
    uniform_logits = outputs.get("uniform_logit", outputs["all_logits"].mean(dim=1)).detach().cpu()
    acc["labels"].extend(labels_np.tolist())
    acc["logits"]["final"].extend(outputs["final_logit"].detach().cpu().reshape(-1).tolist())
    acc["logits"]["base"].extend(outputs["base_logit"].detach().cpu().reshape(-1).tolist())
    acc["logits"]["artifact"].extend(outputs["artifact_logit"].detach().cpu().reshape(-1).tolist())
    acc["logits"]["structure"].extend(outputs["structure_logit"].detach().cpu().reshape(-1).tolist())
    acc["logits"]["patch"].extend(outputs["patch_logit"].detach().cpu().reshape(-1).tolist())
    acc["logits"]["uniform"].extend(uniform_logits.reshape(-1).tolist())
    acc["gate_probs"].append(outputs["gate_probs"].detach().cpu())
    acc["all_logits"].append(all_logits)
    for key in acc["base_family"].keys():
        acc["base_family"][key].extend(outputs["base_family_logits"][key].detach().cpu().reshape(-1).tolist())
    acc["tile_entropy"].extend(outputs["tile_entropy"].detach().cpu().reshape(-1).tolist())
    acc["tile_variance"].extend(outputs["tile_variance"].detach().cpu().reshape(-1).tolist())


def _merge_accumulators(accumulators):
    merged = _empty_accumulator()
    for acc in accumulators:
        merged["labels"].extend(acc["labels"])
        for key in merged["logits"].keys():
            merged["logits"][key].extend(acc["logits"][key])
        for key in merged["base_family"].keys():
            merged["base_family"][key].extend(acc["base_family"][key])
        merged["gate_probs"].extend(acc["gate_probs"])
        merged["all_logits"].extend(acc["all_logits"])
        merged["tile_entropy"].extend(acc["tile_entropy"])
        merged["tile_variance"].extend(acc["tile_variance"])
    return merged


def _analyze_accumulator(acc, args):
    labels = np.asarray(acc["labels"], dtype=np.float32)
    logits_np = {key: np.asarray(value, dtype=np.float32) for key, value in acc["logits"].items()}
    probs_np = {key: 1.0 / (1.0 + np.exp(-value)) for key, value in logits_np.items()}
    metrics = {key: _safe_metrics(labels, probs) for key, probs in probs_np.items()}

    gate_probs = torch.cat(acc["gate_probs"], dim=0) if acc["gate_probs"] else torch.empty(0, 4)
    all_logits = torch.cat(acc["all_logits"], dim=0) if acc["all_logits"] else torch.empty(0, 4)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    if all_logits.numel() > 0:
        target_all = labels_t.unsqueeze(1).expand_as(all_logits)
        risks = torch.nn.functional.binary_cross_entropy_with_logits(all_logits, target_all, reduction="none")
        oracle_winner = risks.argmin(dim=1)
        gate_top1 = gate_probs.argmax(dim=1)
        all_correct = ((all_logits.sigmoid() > 0.5).float() == target_all).bool()
        base_correct = all_correct[:, 0]
        oracle_any_correct = all_correct.any(dim=1)
        base_error = ~base_correct
        gate_match = (gate_top1 == oracle_winner).float().mean().item()
        oracle_counts = torch.bincount(oracle_winner, minlength=4).tolist()
        gate_counts = torch.bincount(gate_top1, minlength=4).tolist()
        oracle_any_rate = oracle_any_correct.float().mean().item()
        if base_error.any():
            base_error_any = (oracle_any_correct[base_error]).float().mean().item()
        else:
            base_error_any = 0.0
    else:
        gate_match = 0.0
        oracle_counts = [0, 0, 0, 0]
        gate_counts = [0, 0, 0, 0]
        oracle_any_rate = 0.0
        base_error_any = 0.0

    base_family = {key: np.asarray(value, dtype=np.float32) for key, value in acc["base_family"].items()}
    hard_fake = (labels == 1) & (logits_np["base"] < args.cie_hard_fake_threshold)
    if np.any(hard_fake):
        spec = np.stack([logits_np["artifact"], logits_np["structure"], logits_np["patch"]], axis=1)
        correction_orig = spec - logits_np["base"][:, None]
        hard_fake_stats = {
            "hard_fake_count": int(hard_fake.sum()),
            "mean_best_specialist_correction_orig_base": _as_float(correction_orig[hard_fake].max(axis=1).mean()),
            "mean_artifact_correction_family_base": _as_float((logits_np["artifact"] - base_family["artifact"])[hard_fake].mean()),
            "mean_structure_correction_family_base": _as_float((logits_np["structure"] - base_family["structure"])[hard_fake].mean()),
            "mean_patch_correction_family_base": _as_float((logits_np["patch"] - base_family["patch"])[hard_fake].mean()),
        }
    else:
        hard_fake_stats = {
            "hard_fake_count": 0,
            "mean_best_specialist_correction_orig_base": 0.0,
            "mean_artifact_correction_family_base": 0.0,
            "mean_structure_correction_family_base": 0.0,
            "mean_patch_correction_family_base": 0.0,
        }

    tile_entropy = np.asarray(acc["tile_entropy"], dtype=np.float32)
    tile_variance = np.asarray(acc["tile_variance"], dtype=np.float32)
    fake_mask = labels == 1
    if np.any(fake_mask):
        fake_low_entropy = _as_float((tile_entropy[fake_mask] < args.cie_patch_entropy_min).mean())
    else:
        fake_low_entropy = 0.0
    patch_stats = {
        "mean_tile_entropy": _as_float(tile_entropy.mean()) if tile_entropy.size else 0.0,
        "mean_tile_variance": _as_float(tile_variance.mean()) if tile_variance.size else 0.0,
        "fake_low_entropy_fraction": fake_low_entropy,
    }

    oracle_stats = {
        "oracle_any_correct": _as_float(oracle_any_rate),
        "base_error_but_any_correct_rate": _as_float(base_error_any),
        "gate_top1_vs_oracle_match_rate": _as_float(gate_match),
        "oracle_winner_count_per_hypothesis": {
            name: int(count) for name, count in zip(HYPOTHESIS_NAMES, oracle_counts)
        },
        "gate_top1_usage_per_hypothesis": {
            name: int(count) for name, count in zip(HYPOTHESIS_NAMES, gate_counts)
        },
    }

    return {
        "metrics": metrics,
        "oracle": oracle_stats,
        "hard_fake": hard_fake_stats,
        "patch": patch_stats,
        "num_samples": int(labels.shape[0]),
    }


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    if args.model_variant != "cie_iapl":
        raise ValueError("Diagnostics require --model_variant cie_iapl")
    if not args.pretrained_model:
        raise ValueError("--pretrained_model is required for diagnostics")

    device = torch.device(args.device)
    creator = _dataset_creator(args)
    dataset_vals, selected_subsets = creator.build_dataset("test", selected_subsets=args.test_selected_subsets)
    loaders = {
        name: DataLoader(
            dataset,
            args.evalbatchsize,
            sampler=SequentialSampler(dataset),
            drop_last=False,
            num_workers=args.num_workers,
        )
        for dataset, name in zip(dataset_vals, selected_subsets)
    }

    model = build_model(args).to(device)
    load_info = _load_checkpoint(model, args.pretrained_model)
    print(f"[CIE_DIAG] checkpoint_load={load_info}")
    model.eval()

    domain_accs = {}
    domain_stats = {}
    with torch.no_grad():
        for domain, loader in loaders.items():
            acc = _empty_accumulator()
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                if not hasattr(model, "forward_debug"):
                    raise AttributeError("CIE diagnostics require model.forward_debug(images).")
                outputs = model.forward_debug(images)
                _append_batch(acc, labels, outputs)
            domain_accs[domain] = acc
            domain_stats[domain] = _analyze_accumulator(acc, args)
            print(
                "[CIE_DIAG] {} final_acc={:.4f} base_acc={:.4f} gate_oracle_match={:.4f}".format(
                    domain,
                    domain_stats[domain]["metrics"]["final"]["acc"],
                    domain_stats[domain]["metrics"]["base"]["acc"],
                    domain_stats[domain]["oracle"]["gate_top1_vs_oracle_match_rate"],
                )
            )

    summary = _analyze_accumulator(_merge_accumulators(domain_accs.values()), args)
    summary["checkpoint_load"] = load_info
    summary["domains"] = selected_subsets

    gate_usage = {
        "aggregate": summary["oracle"]["gate_top1_usage_per_hypothesis"],
        "oracle_aggregate": summary["oracle"]["oracle_winner_count_per_hypothesis"],
        "domains": {
            domain: {
                "gate_top1_usage_per_hypothesis": stats["oracle"]["gate_top1_usage_per_hypothesis"],
                "oracle_winner_count_per_hypothesis": stats["oracle"]["oracle_winner_count_per_hypothesis"],
            }
            for domain, stats in domain_stats.items()
        },
    }

    output_dir = args.cie_diagnostics_output or args.output_dir
    if output_dir:
        output_dir = os.path.join(output_dir, args.model_name)
    else:
        output_dir = "results/cie_iapl_diagnostics"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "cie_diag_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "cie_diag_domain_stats.json"), "w") as f:
        json.dump(domain_stats, f, indent=2)
    with open(os.path.join(output_dir, "cie_diag_gate_usage.json"), "w") as f:
        json.dump(gate_usage, f, indent=2)
    print(f"[CIE_DIAG] wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
