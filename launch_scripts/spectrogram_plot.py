import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from beat_this.preprocessing import load_audio, LogMelSpect


def spectrogram_plot(audio_path):
    waveform, sr = load_audio(audio_path, dtype="float32")
    x = torch.tensor(waveform, dtype=torch.float32)

    n_fft = 2048
    hop_length = 441

    # linear spectrogram
    linear_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2,
        normalized="frame_length",
        center=False,
    )
    linear_spec = linear_transform(x)
    linear_spec = torch.log1p(1000 * linear_spec)

    # mel spectrogram
    mel_transform = LogMelSpect(n_mels=128)
    mel_spec = mel_transform(x).T

    # axes coordinates
    waveform_time = np.arange(len(waveform)) / sr
    freqs_linear = np.linspace(0, sr / 2, linear_spec.shape[0])

    # layout
    fig = plt.figure(figsize=(11, 9))
    gs = GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[40, 1],
        height_ratios=[1, 2, 2],
        hspace=0.3,
        wspace=0.12,
        left=0.08,
        right=0.95,
        top=0.95,
        bottom=0.07
    )

    ax_wave = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[1, 0], sharex=ax_wave)
    ax_mel = fig.add_subplot(gs[2, 0], sharex=ax_wave)

    cax_spec = fig.add_subplot(gs[1, 1])
    cax_mel = fig.add_subplot(gs[2, 1])

    # empty top-right cell
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis("off")

    # waveform
    ax_wave.plot(waveform_time, waveform, color="black", linewidth=0.7)
    ax_wave.set_title("Waveform")
    ax_wave.set_ylabel("Amplitude")

    # linear spectrogram
    im0 = ax_spec.imshow(
        linear_spec.numpy(),
        origin="lower",
        aspect="auto",
        extent=[0, waveform_time[-1], freqs_linear[0], freqs_linear[-1]],
    )
    ax_spec.set_title("Log Spectrogram (STFT)")
    ax_spec.set_ylabel("Frequency (Hz)")
    fig.colorbar(im0, cax=cax_spec)

    # mel spectrogram
    im1 = ax_mel.imshow(
        mel_spec.numpy(),
        origin="lower",
        aspect="auto",
        extent=[0, waveform_time[-1], 0, mel_spec.shape[0]],
    )
    ax_mel.set_title("Log Mel Spectrogram (128 bins)")
    ax_mel.set_ylabel("Mel bins")
    ax_mel.set_xlabel("Time (s)")
    fig.colorbar(im1, cax=cax_mel)

    # consistent time limits
    ax_wave.set_xlim(0, waveform_time[-1])

    plt.savefig("plots/spectrogram_figure.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    audio_path = "data/audio/wav/gtzan/blues/blues.00004.wav"
    spectrogram_plot(audio_path)
