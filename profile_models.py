"""
Model Efficiency Profiler — Phase 4
=====================================
Measures and compares efficiency metrics for all 10 architectures:
  - Parameters (M)
  - Model size on disk (MB)
  - FLOPs / GFLOPs (using thop)
  - GPU Warm-up + Inference latency (ms/image)
  - GPU Throughput (images/second)
  - Peak VRAM usage (MB)

Usage:
  python profile_models.py                   # profile all available trained models
  python profile_models.py --all             # profile all architectures (no weights needed)
  python profile_models.py --model resnet18  # profile a single model
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    vgg19, VGG19_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from thop import profile as thop_profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("[WARN] thop not installed. FLOPs will not be measured.")
    print("       Run: pip install thop")

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DROPOUT         = 0.3
WARMUP_RUNS     = 50
MEASURE_RUNS    = 200
INPUT_SHAPE     = (1, 3, 224, 224)

MODEL_REGISTRY = {
    "resnet18"      : ("pneumonia_resnet18.pt",        "resnet18"),
    "resnet50"      : ("pneumonia_resnet50.pt",        "resnet50"),
    "densenet121"   : ("pneumonia_densenet121.pt",     "densenet121"),
    "vgg19"         : ("pneumonia_vgg19.pt",           "vgg19"),
    "efficientnetb0": ("pneumonia_efficientnet_b0.pt", "efficientnetb0"),
    "mobilenetv3"   : ("pneumonia_mobilenetv3.pt",     "mobilenetv3"),
    "convnext_tiny" : ("pneumonia_convnext_tiny.pt",   "convnext_tiny"),
    "deit_tiny"     : ("pneumonia_deit_tiny.pt",       "deit_tiny"),
    "swin_tiny"     : ("pneumonia_swin_tiny.pt",       "swin_tiny"),
    "vit_b_16"      : ("pneumonia_vit_b_16.pt",        "vit_b_16"),
}

DISPLAY_NAMES = {
    "resnet18"      : "ResNet-18",
    "resnet50"      : "ResNet-50",
    "densenet121"   : "DenseNet-121",
    "vgg19"         : "VGG-19",
    "efficientnetb0": "EfficientNet-B0",
    "mobilenetv3"   : "MobileNetV3-Small",
    "convnext_tiny" : "ConvNeXt-Tiny",
    "deit_tiny"     : "DeiT-Tiny",
    "swin_tiny"     : "Swin-Tiny",
    "vit_b_16"      : "ViT-B/16",
}


# ─── Model Builder (mirrors train_model.py) ───────────────────────────────────
def build_model(key: str) -> nn.Module:
    k = key.lower()
    if k == "resnet18":
        m = resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.fc.in_features, 2))
    elif k == "resnet50":
        m = resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.fc.in_features, 2))
    elif k == "densenet121":
        m = densenet121(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.classifier.in_features, 2))
    elif k == "vgg19":
        m = vgg19(weights=None)
        m.classifier[6] = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.classifier[6].in_features, 2))
    elif k == "efficientnetb0":
        m = efficientnet_b0(weights=None)
        m.classifier[1] = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.classifier[1].in_features, 2))
    elif k == "mobilenetv3":
        m = mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.classifier[3].in_features, 2))
    elif k == "convnext_tiny":
        m = convnext_tiny(weights=None)
        m.classifier[2] = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.classifier[2].in_features, 2))
    elif k == "deit_tiny":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm required for deit_tiny")
        m = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=2)
    elif k == "swin_tiny":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm required for swin_tiny")
        m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=2)
    elif k == "vit_b_16":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm required for vit_b_16")
        m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    else:
        raise ValueError(f"Unknown model key: {key}")
    return m


# ─── Profiling helpers ────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> float:
    """Returns trainable parameter count in millions."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return round(total / 1e6, 2)


def get_weight_size_mb(weight_path: Path) -> float:
    """File size of .pt weight file in MB."""
    if weight_path.exists():
        return round(weight_path.stat().st_size / (1024 ** 2), 2)
    return None


def measure_flops(model: nn.Module, device: torch.device) -> tuple:
    """Returns (GFLOPs, MACs_M) using thop."""
    if not THOP_AVAILABLE:
        return None, None
    dummy = torch.randn(*INPUT_SHAPE).to(device)
    model.eval()
    try:
        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        gflops  = round(macs * 2 / 1e9, 2)   # FLOPs = 2 * MACs
        macs_m  = round(macs / 1e6, 2)
        return gflops, macs_m
    except Exception as e:
        print(f"    [WARN] FLOPs measurement failed: {e}")
        return None, None


def measure_latency(model: nn.Module, device: torch.device) -> dict:
    """
    GPU latency measurement using CUDA events for high precision.
    Falls back to time.perf_counter() on CPU.
    Returns dict with mean_ms, std_ms, throughput_fps.
    """
    model.eval()
    dummy = torch.randn(*INPUT_SHAPE).to(device)

    if device.type == "cuda":
        # Warm up
        with torch.no_grad():
            for _ in range(WARMUP_RUNS):
                _ = model(dummy)
        torch.cuda.synchronize()

        # Measure
        times = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(MEASURE_RUNS):
                start_event.record()
                _ = model(dummy)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))  # ms

    else:
        # CPU timing
        with torch.no_grad():
            for _ in range(WARMUP_RUNS):
                _ = model(dummy)
        times = []
        with torch.no_grad():
            for _ in range(MEASURE_RUNS):
                t0 = time.perf_counter()
                _ = model(dummy)
                times.append((time.perf_counter() - t0) * 1000)

    import statistics
    mean_ms = round(statistics.mean(times), 3)
    std_ms  = round(statistics.stdev(times), 3)
    fps     = round(1000 / mean_ms, 1)

    return {"mean_ms": mean_ms, "std_ms": std_ms, "throughput_fps": fps}


def measure_vram(model: nn.Module, device: torch.device) -> float:
    """Peak VRAM usage during inference in MB."""
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.randn(*INPUT_SHAPE).to(device)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)
    return round(peak / (1024 ** 2), 1)


# ─── Single model profile ─────────────────────────────────────────────────────
def profile_model(model_key: str, load_weights: bool = True) -> dict:
    weight_file, _ = MODEL_REGISTRY[model_key]
    weight_path = BASE_DIR / weight_file
    display     = DISPLAY_NAMES[model_key]

    print(f"\n  Profiling: {display}")
    print(f"  {'─' * 50}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    try:
        model = build_model(model_key)
    except Exception as e:
        print(f"    [ERROR] Cannot build model: {e}")
        return {"model": display, "error": str(e)}

    # Load weights if available
    weights_loaded = False
    if load_weights and weight_path.exists():
        try:
            state = torch.load(str(weight_path), map_location=device)
            model.load_state_dict(state)
            weights_loaded = True
        except Exception as e:
            print(f"    [WARN] Could not load weights: {e}")

    model.to(device)
    model.eval()

    # Parameter count
    params_m = count_parameters(model)
    print(f"    Parameters   : {params_m} M")

    # File size
    size_mb = get_weight_size_mb(weight_path)
    if size_mb:
        print(f"    Weight file  : {size_mb} MB")
    else:
        print(f"    Weight file  : not found (profiling architecture only)")

    # FLOPs
    gflops, macs_m = measure_flops(model, device)
    if gflops:
        print(f"    GFLOPs       : {gflops}")

    # Latency
    print(f"    Measuring latency ({MEASURE_RUNS} runs)...", end=" ", flush=True)
    latency = measure_latency(model, device)
    print(f"{latency['mean_ms']} ms/img  ({latency['throughput_fps']} fps)")

    # VRAM
    vram_mb = measure_vram(model, device)
    if vram_mb:
        print(f"    Peak VRAM    : {vram_mb} MB")

    result = {
        "model"          : display,
        "model_key"      : model_key,
        "weights_loaded" : weights_loaded,
        "params_M"       : params_m,
        "weight_size_MB" : size_mb,
        "GFLOPs"         : gflops,
        "MACs_M"         : macs_m,
        "latency_mean_ms": latency["mean_ms"],
        "latency_std_ms" : latency["std_ms"],
        "throughput_fps" : latency["throughput_fps"],
        "peak_vram_MB"   : vram_mb,
        "device"         : str(device),
    }
    return result


# ─── Print comparison table ───────────────────────────────────────────────────
def print_table(results: list):
    sep  = "─" * 110
    hdr  = (f"{'Model':<18} {'Params(M)':>10} {'Size(MB)':>10} {'GFLOPs':>8} "
            f"{'Latency(ms)':>13} {'Throughput':>12} {'VRAM(MB)':>10}")

    print(f"\n\n{'='*110}")
    print(" Efficiency Profiling Results — All Architectures")
    print(f"{'='*110}")
    print(hdr)
    print(sep)

    for r in results:
        if "error" in r:
            print(f"  {r['model']:<16}  ERROR: {r['error']}")
            continue
        name    = r.get("model",           "N/A")
        params  = f"{r.get('params_M','?')} M"
        size    = f"{r['weight_size_MB']} MB" if r.get('weight_size_MB') else "—"
        gflops  = f"{r.get('GFLOPs','—')}" if r.get('GFLOPs') else "—"
        lat     = f"{r.get('latency_mean_ms','?')} ms"
        fps     = f"{r.get('throughput_fps','?')} fps"
        vram    = f"{r.get('peak_vram_MB','—')} MB" if r.get('peak_vram_MB') else "—"
        print(f"  {name:<16} {params:>10} {size:>10} {gflops:>8} {lat:>13} {fps:>12} {vram:>10}")

    print(sep)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Model Efficiency Profiler")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Profile a single model.")
    parser.add_argument("--all", action="store_true",
                        help="Profile all 10 architectures (no weights required).")
    args = parser.parse_args()

    print("=" * 60)
    print("  Pneumonia Detection — Model Efficiency Profiler (Phase 4)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {torch.cuda.get_device_name(0)} | VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Runs   : {WARMUP_RUNS} warm-up + {MEASURE_RUNS} measurement runs")
    print(f"  Input  : {INPUT_SHAPE}")

    # Determine which models to profile
    if args.model:
        keys_to_profile = [args.model]
        load_weights    = True
    elif args.all:
        keys_to_profile = list(MODEL_REGISTRY.keys())
        load_weights    = False
    else:
        # Default: only models with existing .pt files
        keys_to_profile = [k for k, (f, _) in MODEL_REGISTRY.items()
                           if (BASE_DIR / f).exists()]
        load_weights    = True
        if not keys_to_profile:
            print("\n  No trained models found. Running architecture-only profiling (--all).")
            keys_to_profile = list(MODEL_REGISTRY.keys())
            load_weights    = False

    print(f"\n  Profiling {len(keys_to_profile)} model(s)...")

    results = []
    for key in keys_to_profile:
        r = profile_model(key, load_weights=load_weights)
        results.append(r)

    print_table(results)

    # Save to JSON
    out_path = BASE_DIR / "profiling_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
