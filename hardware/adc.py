"""ADC 模数转换器缺陷模型 — CIR 包络量化近似。

═══════════════════════════════════════════════════════════════════════════
误差层级: 芯片级 (Chip-level) — 发生在接收机 ADC 内部
═══════════════════════════════════════════════════════════════════════════
不属于信道问题，不属于算法问题，是硅片本身的分辨率限制。

═══════════════════════════════════════════════════════════════════════════
物理原理
═══════════════════════════════════════════════════════════════════════════
ADC (Analog-to-Digital Converter) 将天线收到的连续模拟电压/IQ 采样映射为离散
数字码字。一个 N-bit ADC 只能区分 2^N 个不同的幅度档位。

  模拟输入           ADC              数字输出
  ─────────         ─────            ─────────
  0.000 V     →                      000000  (0)
  0.005 V     →     6-bit           000001  (1)
  0.010 V     →     ─────────        000010  (2)
  ...              Δ = Vfs/64
  3.300 V     →                      111111  (63)

量化台阶:  Δ = V_fullscale / (2^N - 1)

每个采样值的真实幅度被"四舍五入"到最近的量化档位:

  x_quantized = round(x / Δ) × Δ
  量化误差:  e = x_quantized - x  ∈ [-Δ/2, +Δ/2]

═══════════════════════════════════════════════════════════════════════════
对测距的影响机制（定位层近似）
═══════════════════════════════════════════════════════════════════════════
严格的 ADC 模型应该作用在复基带 I/Q 或 UWB accumulator 之前。当前项目
重点是测距误差而不是接收机电路，因此这里把 ADC 有效位数近似为
“离散 CIR 包络的动态范围限制”。这个近似用于分析弱首径是否会被量化台阶
淹没，不应解释为完整 ADC 电路仿真。

量化不伤害大信号，但威胁弱信号的存在性:

  CIR 峰值归一化到 1.0:
    Δ = 1.0 / 2^(N-1)          ← 有符号 ADC，最高位是符号位
    6-bit:  Δ = 1/32 ≈ 0.031
    8-bit:  Δ = 1/128 ≈ 0.008
    10-bit: Δ = 1/512 ≈ 0.002
    12-bit: Δ = 1/2048 ≈ 0.0005

  NLOS 场景首径幅度通常为峰值的 2% ~ 15%:
    首径 = 0.05 → 6-bit 下 0.05/0.031 ≈ 1.6 LSB → 保留但有量化偏置
    首径 = 0.02 → 6-bit 下 0.02/0.031 ≈ 0.6 LSB → 量化为 0 → 首径消失

  首径被量化到零 → 检测算法跳到次径 → 测距误差可达若干米。

═══════════════════════════════════════════════════════════════════════════
重要澄清: 相干积累大幅削弱量化影响
═══════════════════════════════════════════════════════════════════════════
UWB 接收机（如 DW1000）在 ADC 采样之后对前导码做上千次相干积累。
积累后等效量化 SNR:

  SNR_quant_eff = 6.02×N + 1.76 + 10×log10(M)

  其中 N = ADC 位数, M = 相干积累次数
  DW1000 (N=6, M=1024): SNR ≈ 68 dB

热噪声 SNR 通常 15~25 dB，远低于 68 dB。
→ 量化噪声在积累后被热噪声完全掩盖，不是主要瓶颈。

但是: 首径和最强径功率比超过 36 dB（6-bit 动态范围）时，
      即使热噪声为零，首径也无法被分辨。

═══════════════════════════════════════════════════════════════════════════
适用的协议 / 传感器
═══════════════════════════════════════════════════════════════════════════
  所有协议均使用 ADC，但受影响程度不同:
  - UWB (DW1000/Decawave): 6-bit → 动态范围窄，最易受影响
  - WiFi (802.11):         10-12 bit → 影响较小
  - 5G NR:                 10-14 bit → 基本不受影响
"""

from __future__ import annotations

import numpy as np


def quantize_cir(
    cir_discrete: np.ndarray,
    adc_bits: int = 6,
) -> np.ndarray:
    """对 CIR 包络施加 ADC 幅度量化近似。

    定位层近似：量化在 CIR 域（通常以 clean CIR 峰值为参考）进行，
    直接模拟设备报告的 CIR 动态范围。这样避免实现完整 I/Q 采样链路，
    但仍能研究量化台阶对首径检测的影响。

    Parameters
    ----------
    cir_discrete : ndarray
        归一化的离散 CIR 包络。
    adc_bits : int
        ADC 有效位数。典型值: UWB=6, WiFi=10, 5G=12。

    Returns
    -------
    ndarray
        量化后的 CIR 包络（幅度被离散化为 2^(adc_bits-1) 个正档位）。
    """
    if adc_bits < 1 or cir_discrete.size == 0:
        return cir_discrete

    n_levels = 2 ** (adc_bits - 1)          # 有符号 ADC 正半轴
    delta = 1.0 / max(n_levels - 1, 1)

    return np.round(cir_discrete / delta) * delta


def quantize_frequency_iq(
    h_freq: np.ndarray,
    adc_bits: int = 6,
) -> np.ndarray:
    """Apply ADC amplitude quantization to complex frequency-domain samples.

    Unlike ``quantize_cir`` which quantizes the CIR envelope (a post-IFFT
    magnitude), this function quantizes the real and imaginary parts of the
    complex frequency-domain samples *before* IFFT.  This respects the
    physical signal chain:

      ADC (time-domain I/Q) → FFT → frequency-domain samples
                    or
      H(f) samples (our simulation shortcut)

    Because the IFFT averages over *N* independently quantized frequency
    bins, the resulting CIR has an effective resolution much finer than the
    raw ADC bit count would suggest (processing gain ≈ √N).  This is exactly
    what happens in a real receiver.

    Parameters
    ----------
    h_freq : ndarray, complex
        Frequency-domain channel response (e.g. from ``path_response`` or
        ``observe_frequency_response``).
    adc_bits : int
        ADC effective bits.  Typical: UWB=6, WiFi=10, 5G=12.

    Returns
    -------
    ndarray, complex
        Quantized frequency-domain response (same dtype and shape).
    """
    if adc_bits < 1 or h_freq.size == 0:
        return h_freq

    # Signed ADC: peak-to-peak range covered by 2^bits levels.
    max_abs = float(max(np.max(np.abs(h_freq.real)), np.max(np.abs(h_freq.imag))))
    if max_abs <= 1e-30:
        return h_freq

    n_levels = 2 ** (adc_bits - 1)          # positive half of signed range
    delta = max_abs / max(n_levels - 1, 1)

    real_q = np.round(h_freq.real / delta) * delta
    imag_q = np.round(h_freq.imag / delta) * delta
    return (real_q + 1j * imag_q).astype(h_freq.dtype)
