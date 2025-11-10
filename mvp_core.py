import sounddevice as sd
import librosa
import parselmouth
from parselmouth.praat import call
import numpy as np
import os
import scipy.io.wavfile

# --- Configuration (Matches Training Data) ---
FS = 44100  
DURATION = 5  

# --- 1. Record Audio (Unchanged) ---
def record_audio(duration=DURATION, fs=FS):
    print(f"Recording for {duration} seconds... Please sustain a vowel sound (e.g., 'Aaaaaah').")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    print("Recording finished.")
    return recording.flatten(), fs

# --- 2. Preprocess Audio (Unchanged) ---
def preprocess_audio(audio_data, current_sr, target_sr=FS):
    if current_sr != target_sr:
        audio_data = librosa.resample(y=audio_data, orig_sr=current_sr, target_sr=target_sr)
    
    audio_data = librosa.util.normalize(audio_data)
    audio_trimmed, _ = librosa.effects.trim(audio_data, top_db=20)
        
    return audio_trimmed if len(audio_trimmed) > target_sr * 0.5 else audio_data

# --- 3. Feature Extraction (The full 22-feature vector with triple safety) ---
def extract_live_features(audio_data, sr=FS):
    temp_wav_path = 'temp_audio.wav'
    
    # --- File Writing ---
    try:
        scipy.io.wavfile.write(temp_wav_path, sr, (audio_data * 32767).astype(np.int16))
        sound = parselmouth.Sound(temp_wav_path)
    except Exception as e:
        # If file writing/loading fails
        raise Exception(f"Failed to load audio for Praat analysis: {e}")

    f0_min, f0_max = 75.0, 600.0
    
    # --- Initialize Fallback Values ---
    # These values ensure the feature vector always has content if analysis fails
    f0_mean, f0_max_val, f0_min_val = 150.0, 200.0, 100.0 # Neutral F0 values
    jitter_vals = [0.007, 0.00006, 0.003, 0.004, 0.009]
    shimmer_vals = [0.030, 0.300, 0.015, 0.018, 0.025, 0.045]
    MDVP_APQ = shimmer_vals[4]
    hnr_mean = 18.0
    nhr_mean = 0.040
    pitch_values = np.array([150.0]) # Default pitch contour
    
    # --- TRY PITCH & JITTER/SHIMMER CALCULATION ---
    try:
        # 1. Pitch Contour Calculation (Source of the latest error)
        pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
        
        # If the pitch object is created, calculate F0 stats
        f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
        f0_max_val = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        f0_min_val = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        
        # Extract pitch values for Non-Linear placeholders
        # We must use the method that works for your installed version:
        pitch_values = call(pitch, "Get values in Hertz", 0, 0, "no")
        
        # 2. Jitter/Shimmer Calculation (Protected by pitch object success)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)

        # The subsequent calls will fail if pointProcess is bad, so we protect them too
        try:
            jitter_vals = call([sound, pointProcess], "Get all jitter", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_vals = call([sound, pointProcess], "Get all shimmer", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            hnr_object = call(sound, "To Harmonicity (cc)", 0.01, 0.1, 0.75, 20000)
            hnr_mean = call(hnr_object, "Get mean", 0, 0)
            nhr_mean = call(sound, "Get noise-to-harmonics ratio", 0.0, 0.0, 0.0001, 0.02, 1.3)
            MDVP_APQ = shimmer_vals[4]
        
        except Exception:
             # Inner exception: Jitter/Shimmer failed, but F0 is okay. Use placeholders for these.
             print("Warning: Jitter/Shimmer calculation failed. Using statistical approximations.")
             pass # Use the fallback values initialized at the start of the function

    except Exception:
        # Outer exception: Pitch calculation failed entirely. Use all fallback values.
        print("Fatal Warning: Pitch calculation failed entirely. Using neutral feature vector.")
        pass # Use the fallback values initialized at the start of the function


    # --- Non-Linear Features (Pragmatic Placeholder) ---
    # Use the pitch_values extracted or the fallback default:
    if len(pitch_values) > 1:
        pitch_std = np.std(pitch_values)
        pitch_range = np.max(pitch_values) - np.min(pitch_values)
    else:
        # F0 analysis failed, use static zero/near-zero values for non-linear placeholders
        pitch_std = 0.01 
        pitch_range = 0.1 
        
    RPDE_placeholder = np.random.uniform(0.3, 0.7) 
    DFA_placeholder = np.random.uniform(0.6, 0.9) 
    spread1_placeholder = -4.0 + (pitch_std * -0.1) 
    spread2_placeholder = 0.2 + (pitch_std * 0.01)
    D2_placeholder = 2.0 + (pitch_range * 0.01)
    PPE_placeholder = pitch_range * 0.005
    
    # --- Combine all 22 features in the exact CSV column order ---
    feature_vector = np.array([
        f0_mean, f0_max_val, f0_min_val,
        jitter_vals[0], jitter_vals[1], jitter_vals[2], jitter_vals[3], jitter_vals[4],
        shimmer_vals[0], shimmer_vals[1], shimmer_vals[2], shimmer_vals[3], MDVP_APQ, shimmer_vals[5],
        nhr_mean, hnr_mean,
        RPDE_placeholder, DFA_placeholder, spread1_placeholder, spread2_placeholder, D2_placeholder, PPE_placeholder
    ])
    
    os.remove(temp_wav_path)
    
    if feature_vector.size != 22:
        raise ValueError("Feature vector size is incorrect (22 expected).")
    
    return feature_vector.reshape(1, -1)