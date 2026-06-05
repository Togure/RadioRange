"""时钟频率误差模型 — SFO (采样频率偏移) 与 CFO (载波频率偏移)。

═══════════════════════════════════════════════════════════════════════════
误差层级: 芯片级 (Chip-level) — 晶体振荡器
═══════════════════════════════════════════════════════════════════════════
所有数字通信系统都依赖晶体振荡器 (Crystal Oscillator) 提供时钟基准。
TX 和 RX 各自有一个独立的晶体，两个晶体的振荡频率不可能完全一致。
这种频率差异体现在两个层面:

  (1) SFO — 采样频率偏移: ADC/DAC 的采样时钟频率不一致
  (2) CFO — 载波频率偏移:   RF 混频器的本振频率不一致

═══════════════════════════════════════════════════════════════════════════
SFO (Sampling Frequency Offset) — 采样频率偏移
═══════════════════════════════════════════════════════════════════════════

物理原理
────────
两个设备的 ADC / DAC 时钟由各自的晶体驱动。TX 的 DAC 以 f_s 采样，
RX 的 ADC 以 f_s' = f_s × (1 + sfo) 采样，其中 sfo 的典型值是 ±20 ppm。

  ppm (parts per million):  1 ppm = 10^-6 的相对频率偏差
  ±20 ppm 的晶体:            实际频率 = 标称值 × (1 ± 20/10^6)

对时间轴的影响:

  发射端时间网格:  t_k = k / f_s
  接收端时间网格:  t'_k = k / (f_s × (1 + sfo_ppm/1e6))

  等效于所有路径延迟被缩放:

    tau' = tau × (1 + sfo_ppm / 1e6)

  sfo > 0 (RX 时钟偏快): tau 变长 → 测得的距离偏大
  sfo < 0 (RX 时钟偏慢): tau 变短 → 测得的距离偏小

  例: sfo = +20 ppm, 真实距离 10m → 测得 10.0002m (误差 0.2mm，可忽略)
  例: sfo = 未补偿, 距离 100m → 测得 100.002m (误差 2mm，仍可忽略)

  但是！对于 OFDM 系统，SFO 还有二次效应:
  - 子载波间距偏移 → 子载波间干扰 (ICI)
  - 导频子载波的相位随子载波索引线性旋转 → 信道估计偏置

对测距的影响机制
────────────────
  UWB (脉冲):
    CIR 时间轴被整体缩放 → 首径 ToF 偏移
    影响: 量级极小 (ppm 级)，对单次测距基本不可见
    累积效应: 若干次测距后无影响（因为误差是比例误差，不是固定偏差）

  WiFi / 5G NR (OFDM):
    SFO 引起符号定时漂移 (symbol timing drift):
    每个 OFDM 符号的 FFT 窗口略有偏移 → 等效于线性相位旋转
    通过导频可以估计并补偿 SFO → 残留 SFO 通常 < 0.1 ppm
    残留 SFO < 0.1 ppm → 对测距影响 < 0.01mm → 完全可忽略

═══════════════════════════════════════════════════════════════════════════
CFO (Carrier Frequency Offset) — 载波频率偏移
═══════════════════════════════════════════════════════════════════════════

物理原理
────────
接收机的本地振荡器 (LO) 频率与发射机的载波频率不完全一致，差异为 Δf Hz。

  TX:  发射信号 = s(t) × exp(j·2π·f_c·t)
  RX:  本振信号  = exp(-j·2π·(f_c + Δf)·t)
  下变频后:      = s(t) × exp(-j·2π·Δf·t)

每条多径分量经历不同的传播延迟 τ_i，因此积累不同的 CFO 相位:

  Δφ_i = -2π · Δf · τ_i

  其中 τ_i 是第 i 条径的绝对单程传播延迟 (含同步偏差)。

  短径 (LOS, τ ≈ 33ns @ 10m): Δφ 小
  长径 (NLOS, τ ≈ 100ns @ 30m): Δφ 更大

关键: CFO 对各径的相位旋转是不同的！因为 τ_i 不同。
      这不是一个公共相位旋转（公共相位可以被估计并补偿）。

对测距的影响机制
────────────────
  UWB (脉冲无线电):
    每条径的复数增益被 CFO 旋转不同的角度。
    CIR 包络 = |Σ a_i × exp(j·φ_i) × exp(-j·2π·Δf·τ_i)|
    由于各径的相位旋转不同 → CIR 包络形状改变 → 首径检测点偏移。

    DW1000 芯片内置 CFO 估计与补偿（基于前导码互相关）。
    残留 CFO < 1 kHz → 对 10m 距离 τ=67ns，相位旋转 < 0.0004 rad → 可忽略。

  WiFi / 5G NR (OFDM):
    CFO 在频域表现为:
    (1) 公共相位旋转 (Common Phase Error, CPE):
        所有子载波旋转相同角度 → 可通过导频估计并补偿
    (2) 子载波间干扰 (Inter-Carrier Interference, ICI):
        CFO 破坏了子载波正交性 → 每个子载波的能量泄漏到相邻子载波
        → 等效 SNR 降低

    WiFi (802.11):   前导码中的短/长训练序列用于 CFO 估计
                     残留 CFO < 1 kHz → 影响小
    5G NR:           TRS (Tracking Reference Signal) 持续跟踪 CFO
                     残留 CFO < 100 Hz → 基本无影响

═══════════════════════════════════════════════════════════════════════════
本实现的简化模型
═══════════════════════════════════════════════════════════════════════════

  SFO: tau' = tau × (1 + sfo_ppm / 1e6)
       在 ChannelTruth 层面作用于 tau_paths_s，所有 radio 共享同一 SFO。

  CFO: a' = a × exp(-j · 2π · cfo_hz · tau_paths_s)
       在 ChannelTruth 层面作用于 a_paths，实现一阶多径相位扰动近似。
       这不是完整 CFO 接收机模型：没有模拟 OFDM 的 ICI、符号间跟踪、
       PLL、或协议前导码补偿。我们这么做是因为本项目关注定位层测距误差，
       只需要把未补偿载波偏差对 CIR 形状的主要影响投影到多径相位上。

  当前实现是一阶模型。二阶效应（ICI、符号定时漂移）可在 radio 的
  observe() 中后续加入。

  与 observation-level 的关系 (两层建模，非重复):

    Truth-level (本文件):
      晶体原始误差 → 物理层多径变形。SFO 缩放 tau_paths_s，CFO 对各径做
      差异化相位旋转。这是”接收机看到什么”——补偿前的真实物理损伤。

    Observation-level (base_radio.observe_frequency_response):
      接收机补偿后的残差。SFO 残差 → 频率域线性相位斜坡；公共相位抖动 →
      所有子载波同步旋转。这是”补偿后还剩什么”。

    两层同时开启 → 完整链路: 原始时钟误差 → 接收机估计/补偿 → 残留误差。
    两层各自独立、作用在不同层级、物理含义不同，不存在重复建模。

═══════════════════════════════════════════════════════════════════════════
适用的协议 / 传感器
═══════════════════════════════════════════════════════════════════════════
  所有协议均受 SFO/CFO 影响，但影响程度和补偿能力不同:

  - UWB (DW1000/Decawave):
      晶体: 典型 ±20 ppm → SFO 比例误差 < 0.002%
      芯片内置 CFO 估计与补偿 → 残留 CFO < 1 kHz
      一阶影响非常小，可忽略

  - WiFi (802.11 a/g/n/ac/ax):
      晶体: 典型 ±20 ppm → 前导码估计并补偿
      残留 SFO < 0.1 ppm (通过导频跟踪)
      残留 CFO < 1 kHz (通过 STF/LTF 估计)
      一阶影响可忽略

  - 5G NR (FR1/FR2):
      晶体: 典型 ±0.5 ~ ±2 ppm (基站级 OCXO) 或 ±10 ppm (UE 级 TCXO)
      TRS 持续跟踪 → 残留 CFO < 100 Hz
      基本不受影响

  例外: 低成本 IoT 设备可能使用 ±50 ppm 或更差的晶体，影响会更大。

═══════════════════════════════════════════════════════════════════════════
配置参数
═══════════════════════════════════════════════════════════════════════════
  impairments:
    enable_sfo: false       # 是否启用采样频率偏移
    sfo_ppm: 0.0            # SFO 值 (ppm)，典型 ±20

    enable_cfo: false       # 是否启用载波频率偏移
    cfo_hz: 0.0             # CFO 值 (Hz)，典型 100~10000
"""

from __future__ import annotations

import numpy as np


def apply_sfo(
    tau_paths_s: np.ndarray,
    sfo_ppm: float,
) -> tuple[np.ndarray, float]:
    """施加采样频率偏移 (SFO) — 缩放所有路径延迟。

    物理: RX 的 ADC 时钟与 TX 的 DAC 时钟频率不一致。
    RX 时钟偏快 (sfo>0) → 采样间隔更短 → 等效延迟变长。
    RX 时钟偏慢 (sfo<0) → 采样间隔更长 → 等效延迟变短。

    Parameters
    ----------
    tau_paths_s : ndarray
        标称绝对单程 TOF [s]。
    sfo_ppm : float
        采样频率偏移 (ppm)。正 = RX 时钟偏快。

    Returns
    -------
    scaled_tau : ndarray
        tau_paths_s × (1 + sfo_ppm / 1e6)。
    scale : float
        实际应用的缩放因子 (1 + sfo_ppm / 1e6)。
    """
    if sfo_ppm == 0.0:
        return tau_paths_s, 1.0
    scale = 1.0 + sfo_ppm / 1e6
    return tau_paths_s * scale, scale


def apply_cfo(
    a_paths: np.ndarray,
    tau_paths_s: np.ndarray,
    cfo_hz: float,
) -> np.ndarray:
    """施加载波频率偏移 (CFO) — 定位层的一阶多径相位扰动。

    物理: RX 本振与 TX 载波频率相差 Δf。
    每条径因传播延迟 τ_i 不同，在下变频时积累不同的相位误差:
      Δφ_i = -2π · Δf · τ_i

    注意：这不是完整的 CFO 接收机模型。真实 WiFi/5G 中 CFO 主要通过
    CPE/ICI/跟踪残差影响 CSI；这里为了定位层仿真，把未补偿 CFO 的主要
    影响近似投影为各径不同的相位旋转，从而改变 CIR 包络和首径检测。

    Parameters
    ----------
    a_paths : ndarray
        复数路径增益。
    tau_paths_s : ndarray
        各径绝对单程 TOF [s]。
    cfo_hz : float
        载波频率偏移 [Hz]。正 = RX 本振频率偏高。

    Returns
    -------
    rotated_a : ndarray
        a_paths × exp(-j · 2π · cfo_hz · tau_paths_s)。
    """
    if cfo_hz == 0.0:
        return a_paths
    phase = -2.0 * np.pi * cfo_hz * tau_paths_s
    return a_paths * np.exp(1j * phase)
