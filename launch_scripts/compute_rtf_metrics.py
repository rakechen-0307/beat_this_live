import os
import torchaudio
import torch
import time
from tqdm import tqdm
from pathlib import Path

from beat_this.inference import load_model
from beat_this.preprocessing import load_audio, LogMelSpect
from beat_this.model.postprocessor import Postprocessor


def compute_rtf_metrics(checkpoint_path, wav_folder):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    device = "cpu"
    print(f"Using {torch.get_num_threads()} CPU threads.")

    model = load_model(checkpoint_path, device)
    model.eval().to(device)

    spect_transform = LogMelSpect(device=device)
    postprocessor = Postprocessor(type="shifted_causal_local_max")

    total_audio_time = 0.0
    total_runtime = 0.0
    max_seconds = 40

    processed = 0
    failed = 0
    failed_files = []

    max_examples = 100

    wav_files = sorted(Path(wav_folder).rglob("*.wav"))

    for wav in tqdm(wav_files, desc="Processing audio"):
        if processed >= max_examples:
            break
        try:
            waveform, sr = load_audio(wav)

            max_samples = int(sr * max_seconds)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]

            audio_duration = len(waveform) / sr

            x = torch.tensor(waveform, dtype=torch.float32)

            t0 = time.perf_counter()
            with torch.inference_mode():
                spect = spect_transform(x)
                pred = model(spect.unsqueeze(0))
                postp_beat, postp_downbeat = postprocessor(pred["beat"], pred["downbeat"], sclm_lookahead_size=1)
            elapsed = time.perf_counter() - t0

            total_audio_time += audio_duration
            total_runtime += elapsed
            processed += 1

            del waveform, x, spect, pred, postp_beat, postp_downbeat

        except Exception as e:
            failed += 1
            failed_files.append((str(wav), repr(e)))
            tqdm.write(f"[WARNING] Failed on {wav}: {e}")
            continue

    rtf = total_runtime / total_audio_time
    print(f"RTF (CPU): {rtf:.4f}")
    print(f"Processed successfully: {processed}")
    print(f"Failed: {failed}")

    if failed_files:
        print("\nFailed files:")
        for path, err in failed_files:
            print(f" - {path}: {err}")


def convert_mp3_to_wav():
    ''' Utility function to convert mp3 files to wav files, if needed. '''
    audio_folder = "data/audio/mp3/beatles/"
    output_audio_folder = "data/audio/wav/beatles/"
    for file in os.listdir(audio_folder):
        if file.endswith(".mp3"):
            audio_path = os.path.join(audio_folder, file)
            waveform, sr = torchaudio.load(audio_path)
            stem = file.split(".")[0]
            out_filename = f"{stem}.wav"
            output_path = os.path.join(output_audio_folder, out_filename)
            torchaudio.save(output_path, waveform, sr)
            print(f"Converted {file} to wav.")


if __name__ == '__main__':
    wav_folder = "data/audio/wav/gtzan/"
    checkpoint_path = "checkpoints/S0 shift_tolerant_weighted_bce-h128-augTrueTrueTruect=1 cc=1 sw256 shifted-3.ckpt"
    compute_rtf_metrics(checkpoint_path, wav_folder)
