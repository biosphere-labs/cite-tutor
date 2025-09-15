"""GPU Memory Validation Utilities for 4GB VRAM Optimization"""

import torch
import psutil
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings


def get_gpu_info() -> Dict:
    """Get detailed GPU information including memory stats."""
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": []
    }

    if torch.cuda.is_available():
        gpu_info["device_count"] = torch.cuda.device_count()

        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            memory_info = {
                "total": torch.cuda.get_device_properties(i).total_memory,
                "allocated": torch.cuda.memory_allocated(i),
                "cached": torch.cuda.memory_reserved(i),
                "free": torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            }

            gpu_info["devices"].append({
                "id": i,
                "name": device_props.name,
                "memory_mb": memory_info["total"] // (1024**2),
                "free_mb": memory_info["free"] // (1024**2),
                "allocated_mb": memory_info["allocated"] // (1024**2),
                "major": device_props.major,
                "minor": device_props.minor
            })

    return gpu_info


def check_4gb_compatibility() -> Tuple[bool, str]:
    """Check if the system is compatible with 4GB VRAM optimization."""
    gpu_info = get_gpu_info()

    if not gpu_info["available"]:
        return False, "No GPU available. CPU-only mode will be very slow."

    max_memory_mb = max([device["memory_mb"] for device in gpu_info["devices"]])

    if max_memory_mb < 3500:  # Account for system overhead
        return False, f"Maximum GPU memory ({max_memory_mb}MB) is below 4GB threshold"

    if max_memory_mb < 4096:
        warning_msg = f"GPU memory ({max_memory_mb}MB) is close to 4GB limit. Monitor usage carefully."
        warnings.warn(warning_msg)
        return True, warning_msg

    return True, f"GPU memory ({max_memory_mb}MB) is sufficient for 4GB optimization"


def estimate_model_memory(model_name: str, quantization: str = "none") -> int:
    """Estimate memory usage for different models in MB."""
    base_sizes = {
        "distilgpt2": 82,
        "google/flan-t5-small": 60,
        "all-MiniLM-L6-v2": 22,
        "gpt2": 137,
        "microsoft/DialoGPT-small": 117
    }

    base_mb = base_sizes.get(model_name, 200)  # Default estimate

    if quantization == "4bit":
        return int(base_mb * 0.25)
    elif quantization == "8bit":
        return int(base_mb * 0.5)
    else:
        return base_mb


def validate_memory_config(config_path: str = "config/models.yaml") -> Dict:
    """Validate that the model configuration fits within 4GB VRAM."""
    config_file = Path(config_path)
    if not config_file.exists():
        return {"valid": False, "error": f"Config file not found: {config_path}"}

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    total_estimated_mb = 0
    model_estimates = {}

    # Estimate memory for each model
    for model_type, model_config in config.items():
        if model_type == "memory_limits":
            continue

        model_name = model_config.get("model", "unknown")
        quantization = config.get("fine_tuning", {}).get("quantization", "none")

        estimated_mb = estimate_model_memory(model_name, quantization)
        model_estimates[model_type] = {
            "model": model_name,
            "estimated_mb": estimated_mb,
            "quantization": quantization
        }
        total_estimated_mb += estimated_mb

    # Check against limits
    memory_limits = config.get("memory_limits", {})
    max_gpu_mb = memory_limits.get("max_gpu_memory_mb", 4096)
    safety_buffer = memory_limits.get("safety_buffer_mb", 512)

    available_mb = max_gpu_mb - safety_buffer

    result = {
        "valid": total_estimated_mb <= available_mb,
        "total_estimated_mb": total_estimated_mb,
        "available_mb": available_mb,
        "max_gpu_mb": max_gpu_mb,
        "safety_buffer_mb": safety_buffer,
        "model_estimates": model_estimates
    }

    if not result["valid"]:
        result["error"] = f"Estimated memory ({total_estimated_mb}MB) exceeds available ({available_mb}MB)"

    return result


def monitor_memory_usage(device: int = 0) -> Dict:
    """Monitor current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"error": "No GPU available"}

    if device >= torch.cuda.device_count():
        return {"error": f"Device {device} not available"}

    memory_info = {
        "device": device,
        "allocated_mb": torch.cuda.memory_allocated(device) // (1024**2),
        "cached_mb": torch.cuda.memory_reserved(device) // (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) // (1024**2),
        "total_mb": torch.cuda.get_device_properties(device).total_memory // (1024**2)
    }

    memory_info["free_mb"] = memory_info["total_mb"] - memory_info["allocated_mb"]
    memory_info["utilization_percent"] = (memory_info["allocated_mb"] / memory_info["total_mb"]) * 100

    return memory_info


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    else:
        print("No GPU available to clear cache")


def check_gpu_memory():
    """Main function to check GPU memory compatibility and configuration."""
    print("=== GPU Memory Validation ===")

    # Check basic GPU info
    gpu_info = get_gpu_info()
    print(f"GPU Available: {gpu_info['available']}")

    if gpu_info["available"]:
        for device in gpu_info["devices"]:
            print(f"Device {device['id']}: {device['name']} ({device['memory_mb']}MB)")

    # Check 4GB compatibility
    compatible, message = check_4gb_compatibility()
    print(f"4GB Compatible: {compatible}")
    print(f"Message: {message}")

    # Validate configuration
    config_validation = validate_memory_config()
    print(f"Config Valid: {config_validation['valid']}")
    if "error" in config_validation:
        print(f"Config Error: {config_validation['error']}")
    else:
        print(f"Estimated Usage: {config_validation['total_estimated_mb']}MB")
        print(f"Available Memory: {config_validation['available_mb']}MB")

    # Current memory usage
    if gpu_info["available"]:
        current_usage = monitor_memory_usage()
        print(f"Current GPU Usage: {current_usage.get('allocated_mb', 'N/A')}MB")
        print(f"GPU Utilization: {current_usage.get('utilization_percent', 'N/A'):.1f}%")


if __name__ == "__main__":
    check_gpu_memory()