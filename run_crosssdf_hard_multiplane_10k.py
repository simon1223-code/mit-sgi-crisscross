from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from scipy.ndimage import label as connected_components
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

DATA = Path("data/bunny_cross_sections.npz")
OUTPUT = Path("data")
METHOD = "crosssdf_hard_multiplane_10k"
TRAINING_STATE = OUTPUT / f"{METHOD}_training_state.pt"
TRAIN_RES = 64
STEPS = 10_000
BASE_CHANNELS = 16
LEARNING_RATE = 5e-4
MIN_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5
EIKONAL_WEIGHT = 1e-3
MINIMUM_SURFACE_WEIGHT = 5e-2
MINIMUM_SURFACE_BETA = 100.0
LOG_EVERY = 100
PROGRESS_EVERY = 1000
SEED = 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(4, out_channels)
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.layers(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, base):
        super().__init__()
        self.enc1 = Block(in_channels, base)
        self.enc2 = Block(base, 2 * base)
        self.bridge = Block(2 * base, 4 * base)
        self.pool = nn.MaxPool3d(2)
        self.up2 = nn.ConvTranspose3d(4 * base, 2 * base, 2, stride=2)
        self.dec2 = Block(4 * base, 2 * base)
        self.up1 = nn.ConvTranspose3d(2 * base, base, 2, stride=2)
        self.dec1 = Block(2 * base, base)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bridge = self.bridge(self.pool(e2))
        d2 = self.dec2(torch.cat((self.up2(bridge), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.out(d1)


def masked_mean(values, mask):
    return values[mask].mean() if mask.any() else values.sum() * 0.0


def eikonal_loss(prediction, spacing):
    dx = torch.diff(prediction, dim=2)[:, :, :, :-1, :-1] / spacing
    dy = torch.diff(prediction, dim=3)[:, :, :-1, :, :-1] / spacing
    dz = torch.diff(prediction, dim=4)[:, :, :-1, :-1, :] / spacing
    norm = torch.sqrt(dx.square() + dy.square() + dz.square() + 1e-8)
    return (norm - 1).square().mean()


def plane_loss(prediction, plane_sdfs, plane_masks, spacing):
    on_total = prediction.new_zeros(())
    off_total = prediction.new_zeros(())
    for plane_sdf, plane_mask in zip(plane_sdfs, plane_masks):
        plane_sdf = plane_sdf[None, None]
        plane_mask = plane_mask[None, None]
        on_mask = plane_mask & (plane_sdf.abs() <= spacing)
        off_mask = plane_mask & ~on_mask & ((prediction < 0) != (plane_sdf < 0))
        on_total = on_total + masked_mean((prediction - plane_sdf).abs(), on_mask)
        off_total = off_total + masked_mean((prediction - plane_sdf).square(), off_mask)
    count = plane_sdfs.shape[0]
    return on_total / count, off_total / count


def objective(prediction, plane_sdfs, plane_masks, regularization_mask, spacing):
    on_loss, off_loss = plane_loss(prediction, plane_sdfs, plane_masks, spacing)
    eikonal = eikonal_loss(prediction, spacing)
    minimum_surface = torch.exp(-MINIMUM_SURFACE_BETA * prediction.abs())[regularization_mask].mean()
    total = on_loss + off_loss + EIKONAL_WEIGHT * eikonal + MINIMUM_SURFACE_WEIGHT * minimum_surface
    return total, {
        "on": on_loss,
        "off": off_loss,
        "eikonal": eikonal,
        "minimum_surface": minimum_surface,
    }


def sample_surface(field, spacing, count, seed):
    vertices, faces, _, _ = marching_cubes(field, level=0, spacing=(spacing,) * 3)
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(faces), size=count, p=areas / areas.sum())
    triangle = triangles[selected]
    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    return (
        (1 - r1)[:, None] * triangle[:, 0]
        + (r1 * (1 - r2))[:, None] * triangle[:, 1]
        + (r1 * r2)[:, None] * triangle[:, 2]
    )


def mesh_metrics(prediction, target, spacing):
    reference = sample_surface(target, spacing, 12_000, SEED)
    predicted = sample_surface(prediction, spacing, 12_000, SEED + 1)
    pred_to_ref = cKDTree(reference).query(predicted, workers=-1)[0]
    ref_to_pred = cKDTree(predicted).query(reference, workers=-1)[0]
    components = connected_components(
        prediction < 0, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )[1]
    return {
        "chamfer_x100": 50 * (pred_to_ref.mean() + ref_to_pred.mean()),
        "hausdorff_x100": 100 * max(pred_to_ref.max(), ref_to_pred.max()),
        "components": components,
    }


def main():
    assert DATA.is_file(), DATA
    OUTPUT.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = False

    with np.load(DATA) as archive:
        target = torch.from_numpy(archive["sdf3d"].astype(np.float32))[None, None]
        observed_mask = torch.from_numpy(archive["observed_mask"].astype(np.float32))[None, None]
        observed_sdf = torch.from_numpy(archive["observed_sdf2d"].astype(np.float32))[None, None]
        observed_normal = torch.from_numpy(archive["observed_normal"].astype(np.float32))[None]
        u_obs = torch.from_numpy(archive["u_obs"].astype(np.float32))[None, None]
        plane_sdfs = torch.from_numpy(archive["sdf2d"].astype(np.float32))[:, None]
        plane_masks = torch.from_numpy(archive["slice_mask"].astype(np.float32))[:, None]
        grid_axis = archive["grid_axis"].astype(np.float32)

    source_res = target.shape[-1]
    size = (TRAIN_RES,) * 3
    if source_res != TRAIN_RES:
        target = F.interpolate(target, size=size, mode="trilinear", align_corners=True)
        observed_sdf = F.interpolate(observed_sdf, size=size, mode="trilinear", align_corners=True)
        observed_normal = F.interpolate(observed_normal, size=size, mode="trilinear", align_corners=True)
        u_obs = F.interpolate(u_obs, size=size, mode="trilinear", align_corners=True)
        plane_sdfs = F.interpolate(plane_sdfs, size=size, mode="trilinear", align_corners=True)
        if source_res % TRAIN_RES == 0:
            factor = source_res // TRAIN_RES
            observed_mask = F.max_pool3d(observed_mask, factor, factor)
            plane_masks = F.max_pool3d(plane_masks, factor, factor)
        else:
            observed_mask = F.interpolate(observed_mask, size=size, mode="nearest")
            plane_masks = F.interpolate(plane_masks, size=size, mode="nearest")

    observed_sdf *= observed_mask
    observed_normal *= observed_mask
    inputs = torch.cat((observed_mask, observed_sdf, observed_normal, u_obs), dim=1).to(DEVICE)
    target = target.to(DEVICE)
    observed_mask = observed_mask.bool().to(DEVICE)
    observed_sdf = observed_sdf.to(DEVICE)
    u_obs = u_obs.to(DEVICE)
    plane_sdfs = plane_sdfs[:, 0].to(DEVICE)
    plane_masks = plane_masks[:, 0].bool().to(DEVICE)
    spacing = 2.0 / (TRAIN_RES - 1)
    contour_band = (plane_masks & (plane_sdfs.abs() <= spacing)).any(dim=0)[None, None]
    regularization_mask = ~contour_band

    model = UNet3D(inputs.shape[1], BASE_CHANNELS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STEPS, eta_min=MIN_LEARNING_RATE
    )
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None
    history = []
    best_loss = np.inf
    best_step = 0
    best_state = None
    start_step = 1
    if TRAINING_STATE.is_file():
        state = torch.load(TRAINING_STATE, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state["scaler"] is not None:
            scaler.load_state_dict(state["scaler"])
        history = state["history"]
        best_loss = state["best_loss"]
        best_step = state["best_step"]
        best_state = state["best_state"]
        start_step = state["step"] + 1
        print({"resuming_from": state["step"], "best_step": best_step})
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    started = perf_counter()

    for step in range(start_step, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            prediction = u_obs * torch.tanh(model(inputs))
            loss, train_terms = objective(
                prediction, plane_sdfs, plane_masks, regularization_mask, spacing
            )
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        del train_terms, loss, prediction

        if step == 1 or step % LOG_EVERY == 0 or step == STEPS:
            with torch.no_grad(), torch.autocast(
                device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP
            ):
                prediction = u_obs * torch.tanh(model(inputs))
                logged_loss, terms = objective(
                    prediction, plane_sdfs, plane_masks, regularization_mask, spacing
                )
            record = {
                "step": step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "total": float(logged_loss),
                **{name: float(value) for name, value in terms.items()},
            }
            history.append(record)
            if record["total"] < best_loss:
                best_loss = record["total"]
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
            if step % PROGRESS_EVERY == 0:
                print({"step": step, "loss": round(record["total"], 6), "best_step": best_step})
                temporary_state = TRAINING_STATE.with_suffix(".tmp")
                torch.save(
                    {
                        "step": step,
                        "model": {
                            name: value.detach().cpu() for name, value in model.state_dict().items()
                        },
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "history": history,
                        "best_loss": best_loss,
                        "best_step": best_step,
                        "best_state": best_state,
                    },
                    temporary_state,
                )
                temporary_state.replace(TRAINING_STATE)
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = u_obs * torch.tanh(model(inputs))

    absolute_error = (prediction - target).abs()
    slice_excess = F.relu(prediction.abs() - observed_sdf.abs())[observed_mask]
    inside_prediction = prediction < 0
    inside_target = target < 0
    intersection = (inside_prediction & inside_target).sum()
    union = (inside_prediction | inside_target).sum().clamp_min(1)
    near_surface = target.abs() < 2 * spacing
    field_metrics = {
        "mae": float(absolute_error.mean()),
        "surface_mae": float(absolute_error[near_surface].mean()),
        "iou": float(intersection / union),
        "eikonal_error": float(eikonal_loss(prediction, spacing)),
        "slice_slab_excess": float(slice_excess.mean()),
        "slice_slab_violation_rate": float((slice_excess > 1e-5).float().mean()),
        "contour_envelope_max": float(F.relu(prediction.abs() - u_obs).max()),
    }

    prediction_np = prediction.float().cpu().numpy()
    target_np = target.float().cpu().numpy()
    prediction_volume = prediction_np.squeeze()
    target_volume = target_np.squeeze()
    mesh_summary = mesh_metrics(prediction_volume, target_volume, spacing)
    history_frame = pd.DataFrame(history)
    metrics_frame = pd.DataFrame([field_metrics], index=[METHOD])
    mesh_frame = pd.DataFrame([mesh_summary], index=[METHOD])

    np.savez_compressed(
        OUTPUT / f"{METHOD}.npz", target=target_np, **{METHOD: prediction_np}
    )
    history_frame.to_csv(OUTPUT / f"{METHOD}_history.csv", index=False)
    metrics_frame.to_csv(OUTPUT / f"{METHOD}_metrics.csv")
    mesh_frame.to_csv(OUTPUT / f"{METHOD}_mesh_metrics.csv")
    torch.save(
        {
            "method": METHOD,
            "step": best_step,
            "state_dict": best_state,
            "config": {
                "steps": STEPS,
                "learning_rate": LEARNING_RATE,
                "minimum_learning_rate": MIN_LEARNING_RATE,
                "seed": SEED,
            },
        },
        OUTPUT / f"{METHOD}.pt",
    )

    base_metrics = pd.read_csv(OUTPUT / "constraint_metrics.csv", index_col=0)
    base_mesh = pd.read_csv(OUTPUT / "crosssdf_paper_metrics.csv", index_col=0)
    pd.concat([base_metrics.drop(index=METHOD, errors="ignore"), metrics_frame]).to_csv(
        OUTPUT / "constraint_metrics_with_crosssdf_hard_multiplane_10k.csv"
    )
    pd.concat([base_mesh.drop(index=METHOD, errors="ignore"), mesh_frame]).to_csv(
        OUTPUT / "crosssdf_paper_metrics_with_crosssdf_hard_multiplane_10k.csv"
    )

    vertices, faces, _, _ = marching_cubes(
        prediction_volume, level=0, spacing=(spacing,) * 3
    )
    grid_min = float(grid_axis[0])
    trimesh.Trimesh(vertices=vertices + grid_min, faces=faces, process=False).export(
        OUTPUT / f"{METHOD}.ply"
    )

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(history_frame["step"], history_frame["total"], label="total")
    ax.plot(history_frame["step"], history_frame["on"], label="on-contour")
    ax.plot(history_frame["step"], history_frame["off"], label="off-contour")
    ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss")
    ax.set_title(METHOD)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / f"{METHOD}_loss.png", dpi=220, facecolor="white")
    plt.close(fig)

    middle = TRAIN_RES // 2
    prediction_slice = prediction_volume[:, :, middle].T
    target_slice = target_volume[:, :, middle].T
    error_slice = np.abs(prediction_slice - target_slice)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, values, title, cmap, limits in (
        (axes[0], target_slice, "target", "coolwarm", (-0.1, 0.1)),
        (axes[1], prediction_slice, METHOD, "coolwarm", (-0.1, 0.1)),
        (axes[2], error_slice, "absolute error", "magma", (0.0, 0.2)),
    ):
        image = ax.imshow(values, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.contour(target_slice, levels=[0], colors="black", linewidths=1.0)
        if ax is axes[1]:
            ax.contour(prediction_slice, levels=[0], colors="lime", linewidths=1.0)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(OUTPUT / f"{METHOD}_slice_{middle}.png", dpi=220, facecolor="white")
    plt.close(fig)

    print({
        "device": str(DEVICE),
        "seconds": round(elapsed, 1),
        "best_step": best_step,
        **field_metrics,
        **mesh_summary,
    })
    if TRAINING_STATE.is_file():
        TRAINING_STATE.unlink()


if __name__ == "__main__":
    main()
