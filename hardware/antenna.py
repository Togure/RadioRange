"""天线缺陷模型 — 相位中心偏移 (PCV)。

═══════════════════════════════════════════════════════════════════════════
误差层级: 设备级 (Device-level) — 天线本身
═══════════════════════════════════════════════════════════════════════════
这不是芯片问题，不是信道问题，是天线这个物理器件的固有特性。
发射天线和接收天线各贡献一次误差。

═══════════════════════════════════════════════════════════════════════════
物理原理
═══════════════════════════════════════════════════════════════════════════
天线不是理想的各向同性点源。电磁波的有效辐射/接收点（称为"相位中心"）
不是天线的几何中心，而且随频率和入射角变化。

  真实天线 (贴片天线为例):

          入射波 (30° 斜入射)
               ╲
                ╲
      ╭─────────────────╮
      │   贴片天线       │  ← 相位中心不在几何中心
      │     ╳ ← 相位中心 │     偏移了 3~10 mm
      │    ↑             │
      ╰────│─────────────╯
           │  偏移矢量 PCV(θ, φ)

  理想点源: tau = 几何距离 / c
  真实天线: tau = (几何距离 + PCV(θ, φ)) / c

PCV 通常来自暗室校准测量，以天线相位方向图的形式给出:
  PCV(θ, φ, f) = 等效相位中心偏移 (单位: 米)

═══════════════════════════════════════════════════════════════════════════
对测距的影响机制
═══════════════════════════════════════════════════════════════════════════
每条多径分量以不同的入射角到达接收天线 → 每条径获得不同的延迟偏移。

  LOS 径 (θ = 0°, 垂直入射):
    PCV ≈ 0 → 基本无偏移

  LOS 径 (θ = 45°, 斜入射):
    PCV ≈ 3-10 mm → Δτ = 10-30 ps → 测距偏差 3-10 mm

  NLOS 反射径 (θ = 60°):
    PCV ≈ 10-30 mm → 更大偏移

  NLOS 反射径 (θ > 75°):
    PCV 急剧增大 → 可能到 cm 级

全链路 (TX + RX 两端):
  Δτ_total = PCV_tx(θ_departure) / c + PCV_rx(θ_arrival) / c

对测距的影响取决于应用场景:
  - 窄视场角 (< 30°):     PCV < 3mm → 可忽略
  - 中视场角 (30° ~ 60°):  PCV ≈ 3-10mm → 可见但不大
  - 宽视场角 (> 60°):      PCV > 10mm → 厘米级误差 → 不可忽略

═══════════════════════════════════════════════════════════════════════════
本实现的定位层简化模型
═══════════════════════════════════════════════════════════════════════════
本项目关注的是测距/定位误差，而不是完整天线电磁仿真。因此在没有暗室
校准文件的情况下，我们不模拟天线电流分布、极化方向图或阵列互耦，而是
直接把 PCV 表现为每条多径的等效 ToF 偏移。

使用简化的角度相关模型:

  PCV_offset(θ) = pcv_magnitude_m × (1 - cos(θ))
    - 垂直入射 (θ=0°):    offset = 0
    - 斜入射 (θ=45°):     offset ≈ 0.3 × pcv_magnitude_m
    - 水平入射 (θ=90°):   offset = pcv_magnitude_m (最大)

  其中 θ 应该是相对天线 boresight 的入射/出射夹角。当前场景还没有显式
  天线姿态，因此使用 elevation 作为代理量。这个近似适合定位层误差敏感性
  分析，不应解释为真实天线方向图。

  若同时有 AoA 与 AoD，则 RX 与 TX 两端各贡献一半量级的 PCV 近似；
  若只有 AoA，则只模拟 RX 端；若没有角度信息，则随机采样小偏移。

  后续可升级为从 .ant 文件加载真实相位方向图。

═══════════════════════════════════════════════════════════════════════════
适用的协议 / 传感器
═══════════════════════════════════════════════════════════════════════════
  所有协议都使用天线 → 全部受影响:
  - UWB:  频率 3.5-8 GHz，天线通常为单极子或贴片，PCV 较显著
  - WiFi: 频率 2.4/5 GHz，常用贴片天线或偶极子，PCV 中等
  - 5G NR: 频率 < 6 GHz 或 mmWave，FR1 类似 WiFi，FR2 阵列天线 PCV 更复杂

═══════════════════════════════════════════════════════════════════════════
配置参数
═══════════════════════════════════════════════════════════════════════════
  impairments:
    enable_antenna_offset: false     # 是否启用天线相位中心偏移
    antenna_pcv_magnitude_m: 0.003   # PCV 最大偏移量 (m)，典型 3-10mm
"""

from __future__ import annotations

import numpy as np


def apply_antenna_pcv(
    tau_paths_s: np.ndarray,
    aoa_azimuth_deg: np.ndarray | None = None,
    aoa_elevation_deg: np.ndarray | None = None,
    aod_azimuth_deg: np.ndarray | None = None,
    aod_elevation_deg: np.ndarray | None = None,
    pcv_magnitude_m: float = 0.003,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """对每条多径施加天线相位中心偏移 (PCV)。

    定位层近似：不仿真完整天线方向图，而是把 PCV 直接转成每条径
    的等效时延偏移。若有 AoA/AoD，分别近似 RX/TX 端贡献；若没有角度
    信息，则施加一个小的随机等效偏移。

    Parameters
    ----------
    tau_paths_s : ndarray
        各路径的绝对单程 TOF [s]。
    aoa_azimuth_deg : ndarray or None
        各路径的到达方位角 [度]。None 表示无角度信息。
    aoa_elevation_deg : ndarray or None
        各路径的到达仰角 [度]。None 表示无角度信息。
    aod_azimuth_deg : ndarray or None
        各路径的出发方位角 [度]。当前简化模型不直接使用，但保留接口。
    aod_elevation_deg : ndarray or None
        各路径的出发仰角 [度]。None 表示无出射角信息。
    pcv_magnitude_m : float
        PCV 最大偏移量 [m]。典型值: 0.003 ~ 0.010。
    rng : Generator or None
        随机数生成器，用于无角度信息时的随机采样。

    Returns
    -------
    ndarray
        PCV 修正后的路径延迟 [s]。
    """
    n_paths = len(tau_paths_s)
    LIGHT_SPEED_MPS = 299792458.0

    offset_m = np.zeros(n_paths, dtype=float)
    has_angle = False

    if aoa_elevation_deg is not None and len(aoa_elevation_deg) == n_paths:
        # RX 端近似：elevation 被当作相对 boresight 的代理角。
        # 这是定位层模型，用于产生厘米/毫米级等效 ToF 偏差。
        offset_m += 0.5 * _pcv_from_elevation(aoa_elevation_deg, pcv_magnitude_m)
        has_angle = True

    if aod_elevation_deg is not None and len(aod_elevation_deg) == n_paths:
        # TX 端近似：若场景提供 AoD，则把发射端 PCV 也计入。
        offset_m += 0.5 * _pcv_from_elevation(aod_elevation_deg, pcv_magnitude_m)
        has_angle = True

    if not has_angle and rng is not None:
        # 无角度信息 → 各径随机偏移（半正态，均值约 0.3 × pcv_magnitude_m）。
        # 这不是天线物理模型，只是定位层不确定性近似。
        offset_m = np.abs(rng.normal(0.0, pcv_magnitude_m * 0.3, size=n_paths))
    elif not has_angle:
        return tau_paths_s

    return tau_paths_s + offset_m / LIGHT_SPEED_MPS


def _pcv_from_elevation(elevation_deg: np.ndarray, pcv_magnitude_m: float) -> np.ndarray:
    elev_rad = np.deg2rad(elevation_deg)
    cos_factor = np.abs(np.cos(elev_rad))
    return pcv_magnitude_m * (1.0 - cos_factor)
