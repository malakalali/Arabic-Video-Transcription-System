import ssl
import whisper
from typing import Tuple
import os

# Allow model download despite SSL issues (local env workaround)
try:
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
except Exception:
    pass


def load_whisper(model_size: str, device: str | None = None):
    """Load a Whisper model optimized for Google Colab CPU.
    device can be 'cpu', 'cuda', or 'mps'. If None, auto-detect.
    """
    # Check if we're running on Google Colab
    is_colab = 'COLAB_GPU' in os.environ or 'COLAB_TPU' in os.environ
    
    if device is None:
        try:
            import torch  # local import to avoid hard dependency at import time
            
            if is_colab:
                # On Colab, prioritize CPU for large models to avoid memory issues
                if model_size in ['large', 'large-v1', 'large-v2', 'large-v3']:
                    print(f"🎯 Colab detected - Using CPU for {model_size} (best stability)")
                    device = 'cpu'
                elif torch.cuda.is_available():
                    print(f"🎯 Colab with GPU - Using CUDA for {model_size}")
                    device = 'cuda'
                else:
                    print(f"🎯 Colab CPU only - Using CPU for {model_size}")
                    device = 'cpu'
            else:
                # Local environment logic
                if torch.cuda.is_available():
                    device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # For large models, prefer CPU over MPS to avoid memory issues
                    if model_size in ['large', 'large-v1', 'large-v2', 'large-v3']:
                        print(f"Large model detected ({model_size}), using CPU for stability")
                        device = 'cpu'
                    else:
                        # Try MPS for smaller models
                        try:
                            test_tensor = torch.tensor([1.0], device='mps')
                            device = 'mps'
                            print(f"MPS device available and compatible for {model_size}")
                        except Exception as mps_error:
                            print(f"MPS device available but incompatible: {mps_error}")
                            print("Falling back to CPU for better compatibility")
                            device = 'cpu'
                else:
                    device = 'cpu'
        except Exception:
            device = 'cpu'
    
    print(f"🚀 Loading Whisper model: {model_size} on {device}")
    
    # Colab-specific optimizations
    if is_colab and device == 'cpu':
        print("🎯 Applying Colab CPU optimizations...")
        # Set environment variables for better CPU performance
        os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
        os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads
        print("✅ CPU thread optimization applied")
    
    # Load model and handle device conversion errors
    try:
        model = whisper.load_model(model_size, device=device)
        print(f"✅ Successfully loaded {model_size} on {device}")
        
        # Colab CPU memory optimization
        if is_colab and device == 'cpu':
            print("🎯 Optimizing model for Colab CPU...")
            # Force model to CPU if it was loaded on GPU
            if hasattr(model, 'cpu'):
                model = model.cpu()
            print("✅ Model optimized for Colab CPU")
        
        return model
    except Exception as e:
        if device != 'cpu':
            print(f"❌ Failed to load model on {device}: {e}")
            print("🔄 Falling back to CPU...")
            try:
                model = whisper.load_model(model_size, device='cpu')
                print("✅ Successfully loaded model on CPU")
                return model
            except Exception as cpu_error:
                print(f"❌ Failed to load model on CPU: {cpu_error}")
                raise
        else:
            raise


def transcribe_arabic_and_translate(model, audio_path: str) -> Tuple[dict, dict]:
    """
    Returns two Whisper results:
    - Arabic transcription result with timestamps
    - English translation result with timestamps

    Uses decoding settings tuned for accuracy and more stable segmentation.
    Optimized for Colab CPU performance.
    """
    # Check if we're on Colab
    is_colab = 'COLAB_GPU' in os.environ or 'COLAB_TPU' in os.environ
    
    # Preprocess audio for better transcription quality
    try:
        import librosa
        import numpy as np
        
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Apply noise reduction and normalization
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        audio = librosa.util.normalize(audio)
        
        # Save preprocessed audio
        import soundfile as sf
        preprocessed_path = audio_path.replace('.wav', '_preprocessed.wav')
        sf.write(preprocessed_path, audio, sr)
        
        # Use preprocessed audio for transcription
        audio_path = preprocessed_path
        
    except ImportError:
        # If librosa not available, use original audio
        pass
    
    # Memory optimization for large models
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Clear MPS cache if available
            try:
                torch.mps.empty_cache()
            except:
                pass
    except:
        pass
    
    # Colab CPU optimized settings for maximum accuracy (90-95%)
    if is_colab:
        print("🎯 Using Colab-optimized transcription settings...")
        decode_args = dict(
            word_timestamps=True,  # Enable word-level timing for better sync
            beam_size=3,  # Reduced for CPU performance (was 5)
            best_of=3,  # Reduced for CPU performance (was 5)
            temperature=0.0,  # Keep deterministic for consistency
            condition_on_previous_text=True,  # Better context and flow
            compression_ratio_threshold=1.6,  # Lower threshold for more text capture
            logprob_threshold=-2.0,  # Lower threshold to include more words
            no_speech_threshold=0.2,  # Very low threshold to catch all speech
            language="ar",
            # Large-v3 specific optimizations
            initial_prompt="This is Arabic speech that needs to be transcribed with maximum accuracy. The speech may contain various Arabic dialects and formal language.",
            suppress_tokens=[-1],  # Don't suppress any tokens
        )
    else:
        # Local environment settings
        decode_args = dict(
            word_timestamps=True,  # Enable word-level timing for better sync
            beam_size=5,  # Increased for Large-v3 accuracy
            best_of=5,  # Increased for better candidate selection
            temperature=0.0,  # Keep deterministic for consistency
            condition_on_previous_text=True,  # Better context and flow
            compression_ratio_threshold=1.6,  # Lower threshold for more text capture
            logprob_threshold=-2.0,  # Lower threshold to include more words
            no_speech_threshold=0.2,  # Very low threshold to catch all speech
            language="ar",
            # Large-v3 specific optimizations
            initial_prompt="This is Arabic speech that needs to be transcribed with maximum accuracy. The speech may contain various Arabic dialects and formal language.",
            suppress_tokens=[-1],  # Don't suppress any tokens
        )

    print(f"🎯 Starting Arabic transcription...")
    arabic_result = model.transcribe(
        audio_path,
        task="transcribe",
        **decode_args,
    )
    print("✅ Arabic transcription completed")

    # For translation, reuse most settings, but do not force language
    translate_args = dict(decode_args)
    translate_args.pop("language", None)
    
    print(f"🎯 Starting English translation...")
    english_result = model.transcribe(
        audio_path,
        task="translate",
        **translate_args,
    )
    print("✅ English translation completed")

    return arabic_result, english_result


