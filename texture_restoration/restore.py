"""Optimize SH appearance, preserving all geometry and other PLY properties."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch
import torch.nn.functional as F
from gsplat import rasterization, spherical_harmonics

from .loss import RealDataRobustTextureLoss


def _image(path, device):
    with Image.open(path) as image:
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            raise ValueError(f"Composite alpha images onto the chosen background first: {path}")
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255
    return torch.tensor(array, device=device).permute(2, 0, 1)


def _views(manifest, device):
    manifest = Path(manifest).resolve()
    rows = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("View manifest must be a nonempty JSON list")
    views = []
    for row in rows:
        original = _image(manifest.parent / row["original"], device)
        rendered = _image(manifest.parent / row["rendered"], device)
        if original.shape != rendered.shape or min(original.shape[1:]) < 24:
            raise ValueError("Aligned RGB pairs must have equal dimensions, at least 24x24")
        K = torch.tensor(row["K"], dtype=torch.float32, device=device)
        w2c = torch.tensor(row["w2c"], dtype=torch.float32, device=device)
        if K.shape != (3, 3) or w2c.shape != (4, 4):
            raise ValueError("K must be 3x3 and w2c must be 4x4")
        if not torch.isfinite(K).all() or not torch.isfinite(w2c).all():
            raise ValueError("Camera matrices must be finite")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("Focal lengths must be positive")
        expected = w2c.new_tensor([0, 0, 0, 1])
        rotation = w2c[:3, :3]
        if (not torch.allclose(w2c[3], expected, atol=1e-4)
                or not torch.allclose(rotation @ rotation.T, torch.eye(3, device=device), atol=1e-3)
                or not torch.isclose(torch.linalg.det(rotation), w2c.new_tensor(1.), atol=1e-3)):
            raise ValueError("w2c must be a rigid world-to-camera transform")
        views.append((original, rendered, K, w2c))
    return views


def restore_texture(ply_path, views_json, output_path, *, steps=200, lr=3e-4,
                    hf_weight=0.01, background=(0., 0., 0.)):
    """Return output Path. Inputs use standard 3DGS log scales/logit opacity.

    views_json contains original/rendered paths and OpenCV K/w2c matrices.
    Geometry is fixed; differentiable re-rendering updates only SH coefficients.
    """
    if steps < 1 or not math.isfinite(lr) or lr <= 0 or not math.isfinite(hf_weight) or hf_weight < 0:
        raise ValueError("steps/lr must be positive; hf_weight must be nonnegative")
    if len(background) != 3 or not all(math.isfinite(x) and 0 <= x <= 1 for x in background):
        raise ValueError("background must contain three values in [0, 1]")
    output = Path(output_path).resolve()
    if output == Path(ply_path).resolve():
        raise ValueError("Output must differ from input PLY")
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU and CUDA-enabled gsplat/PyTorch are required")
    device = torch.device("cuda")
    views = _views(views_json, device)
    ply = PlyData.read(str(ply_path))
    vertices = ply["vertex"].data
    names = vertices.dtype.names
    required = ["x", "y", "z", "opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)] + [f"f_dc_{i}" for i in range(3)]
    missing = set(required) - set(names)
    if missing or not len(vertices):
        raise ValueError(f"Expected a nonempty 3DGS PLY; missing properties: {sorted(missing)}")
    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    count = len(rest_names) // 3 + 1
    degree = math.isqrt(count) - 1
    if len(rest_names) % 3 or (degree + 1)**2 != count or not 0 <= degree <= 3:
        raise ValueError("Supported SH degrees are 0 through 3")
    if rest_names != [f"f_rest_{i}" for i in range(len(rest_names))]:
        raise ValueError("SH property indices must be contiguous")

    def tensor(fields):
        array = np.stack([vertices[n] for n in fields], axis=-1).astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite PLY values in {fields}")
        return torch.tensor(array, device=device)

    means = tensor(["x", "y", "z"])
    scales = tensor([f"scale_{i}" for i in range(3)]).exp()
    opacity = tensor(["opacity"]).squeeze(-1).sigmoid()
    quats = tensor([f"rot_{i}" for i in range(4)])
    if (quats.norm(dim=-1) < 1e-8).any() or not torch.isfinite(scales).all():
        raise ValueError("Invalid Gaussian rotations or scales")
    quats = F.normalize(quats, dim=-1)
    dc = torch.nn.Parameter(tensor([f"f_dc_{i}" for i in range(3)])[:, None])
    rest = torch.nn.Parameter(tensor(rest_names).reshape(-1, 3, count - 1).transpose(1, 2).contiguous()) if rest_names else None
    parameters = [dc] if rest is None else [dc, rest]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    loss_fn = RealDataRobustTextureLoss().to(device)
    bg = torch.tensor(background, dtype=torch.float32, device=device)[None]

    def render(view):
        original, _, K, w2c = view
        center = torch.linalg.inv(w2c)[:3, 3]
        coeffs = dc if rest is None else torch.cat([dc, rest], dim=1)
        colors = (spherical_harmonics(degree, F.normalize(means - center, dim=-1), coeffs) + .5).clamp(0, 1)
        image, _, _ = rasterization(means=means, quats=quats, scales=scales,
            opacities=opacity, colors=colors, viewmats=w2c[None], Ks=K[None],
            width=original.shape[2], height=original.shape[1], backgrounds=bg, packed=False)
        return image[0].permute(2, 0, 1)

    with torch.no_grad():
        baseline = [float(F.l1_loss(v[1], v[0])) for v in views]
        initial = [float(F.l1_loss(render(v), v[0])) for v in views]
    for step in range(1, steps + 1):
        view = views[(step - 1) % len(views)]
        optimizer.zero_grad(set_to_none=True)
        prediction = render(view)
        low, high, edge, weight, _ = loss_fn(prediction, view[0], step, steps,
            ref_start_step=1, lambda_H_max=hf_weight)
        loss = F.l1_loss(prediction, view[0]) + low + weight * high + loss_fn.lambda_edge * edge
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"step={step}/{steps} loss={loss.item():.6f}")
    with torch.no_grad():
        final = [float(F.l1_loss(render(v), v[0])) for v in views]
    dc_array = dc.detach().cpu().numpy()[:, 0]
    for i in range(3):
        vertices[f"f_dc_{i}"] = dc_array[:, i]
    if rest is not None:
        array = rest.detach().transpose(1, 2).contiguous().cpu().numpy().reshape(len(vertices), -1)
        for i, name in enumerate(rest_names):
            vertices[name] = array[:, i]
    output.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(output))
    print(json.dumps({"provided_render_l1": baseline, "initial_rerender_l1": initial,
                      "final_rerender_l1": final, "output": str(output)}))
    return output
