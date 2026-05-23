# imx8-edge-ai
Edge AI inference benchmarks on NXP iMX8MP — CPU vs NPU using TFLite + Docker

# Edge AI Inference on NXP iMX8MP

Real-world AI inference benchmarks running MobileNetV1 (INT8 quantized)
on NXP iMX8MP — comparing CPU and NPU performance inside Docker containers.

## Hardware
- **Board:** CompuLab IOT-GATE-iMX8Plus
- **SoC:** NXP i.MX8M Plus (Cortex-A53 quad-core + Vivante NPU 2.3 TOPS)
- **OS:** Debian 12 (Bookworm)
- **Kernel:** 6.6.23-compulab

## Benchmark Results — MobileNetV1 1.0 224 INT8

| Method              | Latency  | FPS   | vs Baseline |
|---------------------|----------|-------|-------------|
| CPU — 1 thread      | 175.00ms |   5.7 | 1x          |
| CPU — 4 threads     |  49.85ms |  20.1 | 3.5x        |
| NPU — VX Delegate   |   3.51ms | 284.9 | **50x**     |

## Architecture

### WSL2 (x86 build machine)
- **docker buildx** → ARM64 container image
- **scp** → iMX8MP board
- **docker run** --device /dev/galcore
- **libvx_delegate.so** → Vivante NPU

## Key Technical Decisions

- **Build on WSL2, run on iMX8** — cross-compilation workflow
- **Torizon wayland-base-vivante:3** — NXP Vivante GPU stack pre-installed, no BSP modification needed on host
- **INT8 quantized model** — required for NPU acceleration
- **--device /dev/galcore** — exposes NPU to container

## CPU Inference Container

```bash
cd cpu-inference
docker buildx build --platform linux/arm64 -t edge-inference:cpu .
```

## NPU Inference Container

```bash
cd npu-inference
docker buildx build --platform linux/arm64 -t edge-inference:npu .
docker run --rm --device /dev/galcore --group-add video edge-inference:npu
```

## What I Learned

- NXP Vivante GPU stack: libGAL → libOpenVX → libvx_delegate → TFLite
- Container device passthrough architecture (--device /dev/galcore)
- Kernel driver vs user-space library separation
- Torizon base image contains full NXP user-space GPU stack
- INT8 quantization enables 50x NPU speedup vs CPU baseline
- Cross-compilation: build ARM64 binaries on x86 WSL2
