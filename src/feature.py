import librosa
import numpy as np
import os
import json
from scipy import signal
from scipy.stats import skew, kurtosis


def extract_advanced_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=16000, duration=30)

        features = []

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        for v in mfcc_mean:
            features.append(v)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(np.mean(spectral_bandwidth))
        features.append(np.std(spectral_bandwidth))

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.append(np.mean(spectral_contrast))
        features.append(np.std(spectral_contrast))

        stft = np.abs(librosa.stft(y))
        flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
        features.append(np.mean(flux))
        features.append(np.std(flux))

        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(np.mean(chroma))
        features.append(np.std(chroma))

        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pitch_vals = []

        for t in range(pitches.shape[1]):
            idx = mags[:, t].argmax()
            p = pitches[idx, t]
            if p > 0:
                pitch_vals.append(p)

        if len(pitch_vals) > 0:
            features.append(np.mean(pitch_vals))
            features.append(np.std(pitch_vals))
            features.append(max(pitch_vals))
            features.append(min(pitch_vals))
        else:
            features.append(0)
            features.append(0)
            features.append(0)
            features.append(0)

        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        hnr = np.sum(harmonic * harmonic) / (np.sum(percussive * percussive) + 1e-6)
        features.append(np.log(hnr + 1))

        if len(pitch_vals) > 1:
            diffs = np.abs(np.diff(pitch_vals))
            jitter = np.mean(diffs) / (np.mean(pitch_vals) + 1e-6)
            features.append(jitter)
        else:
            features.append(0)

        amp = np.abs(librosa.stft(y))
        amp_mean = np.mean(amp, axis=0)
        if len(amp_mean) > 1:
            amp_diff = np.abs(np.diff(amp_mean))
            shimmer = np.mean(amp_diff) / (np.mean(amp_mean) + 1e-6)
            features.append(shimmer)
        else:
            features.append(0)

        try:
            spec_mean = np.mean(stft, axis=1)
            peaks, _ = signal.find_peaks(spec_mean, distance=20)
            if len(peaks) >= 3:
                features.append(peaks[0])
                features.append(peaks[1])
                features.append(peaks[2])
            else:
                features.append(0)
                features.append(0)
                features.append(0)
        except:
            features.append(0)
            features.append(0)
            features.append(0)

        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        features.append(np.max(rms) / (np.mean(rms) + 1e-6))

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)

        segments = librosa.effects.split(y, top_db=20)
        if len(segments) > 0:
            voiced = 0
            for s, e in segments:
                voiced += (e - s)
            features.append(voiced / len(y))
            features.append(len(segments))
        else:
            features.append(0)
            features.append(0)

        features.append(skew(y))
        features.append(kurtosis(y))
        features.append(np.median(np.abs(y)))
        features.append(np.percentile(np.abs(y), 95))

        clean_features = []

        for x in features:
         if isinstance(x, np.ndarray):
            clean_features.append(float(x.flatten()[0]))
         else:
            clean_features.append(float(x))

        features = clean_features

        return features

    except Exception as e:
        print("Error:", audio_file, e)
        return None


def extract_all_features():
    data = []
    languages = ["english", "hindi", "tamil", "telugu", "malayalam"]

    for lang in languages:
        human_path = "data/human/" + lang
        if os.path.exists(human_path):
            files = os.listdir(human_path)
            for i in range(len(files)):
                if files[i].endswith(".mp3"):
                    fpath = os.path.join(human_path, files[i])
                    feats = extract_advanced_features(fpath)
                    if feats is not None:
                        data.append({
                            "features": feats,
                            "label": "HUMAN",
                            "language": lang
                        })
                if (i + 1) % 100 == 0:
                    print(lang, "human", i + 1)

        ai_path = "data/ai/" + lang
        if os.path.exists(ai_path):
            files = os.listdir(ai_path)
            for i in range(len(files)):
                if files[i].endswith(".mp3"):
                    fpath = os.path.join(ai_path, files[i])
                    feats = extract_advanced_features(fpath)
                    if feats is not None:
                        data.append({
                            "features": feats,
                            "label": "AI_GENERATED",
                            "language": lang
                        })
                if (i + 1) % 100 == 0:
                    print(lang, "ai", i + 1)

    out = open("advanced_features.json", "w")
    json.dump(data, out)
    out.close()

    print("Total samples:", len(data))
    if len(data) > 0:
        print("Features per sample:", len(data[0]["features"]))

    return data


if __name__ == "__main__":
    extract_all_features()
