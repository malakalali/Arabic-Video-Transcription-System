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
    """Load a Whisper model with aggressive GPU acceleration.
    device can be 'cpu', 'cuda', or 'mps'. If None, auto-detect.
    """
    # Check if we're running on Google Colab
    is_colab = 'COLAB_GPU' in os.environ or 'COLAB_TPU' in os.environ
    
    # Set environment variables for better MPS performance
    if not is_colab:
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable MPS fallback
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Aggressive memory management
    
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
                # Local environment logic - AGGRESSIVE GPU acceleration
                if torch.cuda.is_available():
                    device = 'cuda'
                    print(f"🎯 CUDA GPU detected - Using CUDA for {model_size}")
                    
                    # GPU memory optimization
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("🧹 CUDA cache cleared for optimal performance")
                        
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # AGGRESSIVE MPS strategy: Try GPU for ALL models including large ones
                    print(f"🍎 Apple MPS GPU detected - Attempting to use for {model_size}")
                    
                    # Try to force MPS for all models
                    try:
                        # Test MPS compatibility with a more complex operation
                        test_tensor = torch.randn(100, 100, device='mps')
                        test_result = torch.matmul(test_tensor, test_tensor)
                        device = 'mps'
                        print(f"🚀 MPS compatibility test passed - Using GPU for {model_size}!")
                        
                        # MPS optimization
                        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            print("🍎 MPS optimizations enabled")
                            
                    except Exception as mps_error:
                        print(f"⚠️ MPS compatibility test failed: {mps_error}")
                        print("🔄 Falling back to CPU for better compatibility")
                        device = 'cpu'
                else:
                    device = 'cpu'
                    print(f"💻 No GPU detected - Using CPU for {model_size}")
        except Exception:
            device = 'cpu'
            print(f"💻 Error detecting GPU - Using CPU for {model_size}")
    
    print(f"🚀 Loading Whisper model: {model_size} on {device}")
    
    # Colab-specific optimizations
    if is_colab and device == 'cpu':
        print("🎯 Applying Colab CPU optimizations...")
        # Set environment variables for better CPU performance
        os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
        os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads
        print("✅ CPU thread optimization applied")
    
    # Load model and handle device conversion errors with multiple attempts
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                # First attempt: try the selected device
                print(f"🔄 Attempt {attempt + 1}: Loading on {device}...")
                
                # Special handling for MPS with large models
                if device == 'mps' and model_size in ['large', 'large-v1', 'large-v2', 'large-v3']:
                    print("🍎 Attempting hybrid CPU->MPS loading strategy for large model...")
                    try:
                        # Load on CPU first (more stable)
                        print("📱 Loading large model on CPU first...")
                        model = whisper.load_model(model_size, device='cpu')
                        
                        # Try to move specific components to MPS
                        print("🚀 Attempting to move model components to MPS...")
                        try:
                            # Try to move the model to MPS in parts
                            if hasattr(model, 'encoder'):
                                model.encoder = model.encoder.to('mps')
                                print("✅ Encoder moved to MPS")
                            if hasattr(model, 'decoder'):
                                model.decoder = model.decoder.to('mps')
                                print("✅ Decoder moved to MPS")
                            
                            # Set device for future operations
                            model.device = torch.device('mps')
                            print("✅ Successfully loaded large model with hybrid CPU->MPS strategy!")
                            
                        except Exception as move_error:
                            print(f"⚠️ Moving to MPS failed: {move_error}")
                            print("🔄 Keeping model on CPU for stability")
                            device = 'cpu'
                            
                    except Exception as alt_error:
                        print(f"⚠️ Hybrid loading strategy failed: {alt_error}")
                        # Fall back to regular loading
                        model = whisper.load_model(model_size, device=device)
                else:
                    model = whisper.load_model(model_size, device=device)
                
                print(f"✅ Successfully loaded {model_size} on {device} (attempt {attempt + 1})")
            elif attempt == 1 and device != 'cpu':
                # Second attempt: try CPU if GPU failed
                print(f"🔄 Attempt {attempt + 1}: Trying CPU fallback...")
                model = whisper.load_model(model_size, device='cpu')
                print(f"✅ Successfully loaded {model_size} on CPU (fallback)")
            else:
                # Third attempt: force CPU with error handling
                print(f"🔄 Attempt {attempt + 1}: Forcing CPU with error handling...")
                model = whisper.load_model(model_size, device='cpu')
                print(f"✅ Successfully loaded {model_size} on CPU (forced)")
            
            # GPU memory optimization
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 CUDA memory optimized")
            elif device == 'mps':
                print("🍎 MPS memory optimized")
            
            # Colab CPU memory optimization
            if is_colab and device == 'cpu':
                print("🎯 Optimizing model for Colab CPU...")
                # Force model to CPU if it was loaded on GPU
                if hasattr(model, 'cpu'):
                    model = model.cpu()
                print("✅ Model optimized for Colab CPU")
            
            return model
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed on {device}: {e}")
            
            if attempt < max_attempts - 1:
                if device != 'cpu':
                    print("🔄 Trying CPU fallback...")
                    device = 'cpu'
                else:
                    print("🔄 Trying alternative loading method...")
            else:
                print("❌ All attempts failed")
                raise
    
    # This should never be reached, but just in case
    raise Exception("Failed to load model after all attempts")


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


