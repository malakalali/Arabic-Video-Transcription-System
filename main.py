import os
import ssl
import logging
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import subprocess
import tempfile
import shutil
import traceback

# Import our video processing functions
from src.translation import load_whisper, transcribe_arabic_and_translate
from src.subtitles import (
    create_bilingual_srt,
    load_srt_segments_with_transliteration,
    create_bilingual_srt_from_existing,
    escape_for_ffmpeg_subtitles,
    align_english_to_arabic_segments,
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check if we're running on Google Colab
IS_COLAB = 'COLAB_GPU' in os.environ or 'COLAB_TPU' in os.environ

if IS_COLAB:
    logger.info("🎯 Google Colab environment detected!")
    logger.info("🚀 Applying Colab-specific optimizations...")
    
    # Colab CPU optimizations
    os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads
    os.environ['OPENBLAS_NUM_THREADS'] = '4'  # Limit OpenBLAS threads
    
    # Set Flask to use Colab-friendly settings
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    logger.info("✅ Colab optimizations applied")

# Temporary workaround: allow model download despite SSL issues
try:
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
except Exception:
    pass

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Global variable to store the loaded model
whisper_model = None

def extract_audio(video_path, audio_output_path):
    """Extracts audio from a video file using ffmpeg."""
    try:
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',  # Overwrite output file if it exists
            audio_output_path
        ]
        logger.info(f"Running ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("Audio extraction successful")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Audio extraction error: {e}")
        return False

def burn_subtitles_into_video(video_path, srt_path, output_path):
    """Burns subtitles into video using ffmpeg."""
    try:
        # Escape the SRT path for ffmpeg
        escaped_srt_path = escape_for_ffmpeg_subtitles(srt_path)
        
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f"subtitles={escaped_srt_path}:force_style='Fontsize=24,Fontname=Arial Unicode MS'",
            '-c:a', 'copy',
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        logger.info(f"Running ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("Subtitle burning successful")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Subtitle burning error: {e}")
        return False

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/process-video', methods=['POST'])
def process_video():
    try:
        logger.info("Processing video request received")
        
        if 'video' not in request.files:
            logger.error("No video file in request")
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        model_size = request.form.get('model_size', 'large-v3')  # Default to Large-v3 for maximum accuracy
        
        logger.info(f"Video file: {video_file.filename}, Model size: {model_size}")
        
        if video_file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No video file selected'}), 400
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temp directory: {temp_dir}")
            
            # Save uploaded video
            video_path = os.path.join(temp_dir, secure_filename(video_file.filename))
            video_file.save(video_path)
            logger.info(f"Saved video to: {video_path}")
            
            # Extract audio
            audio_path = os.path.join(temp_dir, 'audio.wav')
            logger.info("Extracting audio...")
            if not extract_audio(video_path, audio_path):
                return jsonify({'error': 'Failed to extract audio from video'}), 500
            
            # Load Whisper model if not already loaded
            global whisper_model
            if whisper_model is None:
                logger.info(f"Loading Whisper model: {model_size}")
                try:
                    whisper_model = load_whisper(model_size)
                    logger.info(f"Whisper model loaded successfully")
                except Exception as model_error:
                    logger.error(f"Failed to load {model_size} model: {model_error}")
                    return jsonify({'error': f'Failed to load Whisper model: {model_size}. Error: {model_error}'}), 500
            
            # Transcribe and translate with tuned decoding
            logger.info("Starting transcription and translation...")
            arabic_result, english_result = transcribe_arabic_and_translate(whisper_model, audio_path)
            logger.info("Transcription and translation completed")
            
            # Create SRT file - ALWAYS use actual video content for accurate results
            srt_path = os.path.join(temp_dir, 'subtitles.srt')
            logger.info("Creating SRT from actual video content for accuracy")
            
            try:
                create_bilingual_srt(arabic_result, english_result, srt_path)
                logger.info("SRT file created successfully from video content")
            except Exception as srt_error:
                logger.error(f"Failed to create SRT file: {srt_error}")
                return jsonify({'error': f'Failed to create subtitle file: {srt_error}'}), 500
            
            # Burn subtitles into video
            output_filename = f"processed_{secure_filename(video_file.filename)}"
            output_path = os.path.join(temp_dir, output_filename)
            logger.info("Burning subtitles into video...")
            if not burn_subtitles_into_video(video_path, srt_path, output_path):
                return jsonify({'error': 'Failed to burn subtitles into video'}), 500
            
            # Copy to output directory
            final_output_dir = 'output_videos'
            os.makedirs(final_output_dir, exist_ok=True)
            final_output_path = os.path.join(final_output_dir, output_filename)
            
            try:
                shutil.copy2(output_path, final_output_path)
                logger.info(f"Video saved to: {final_output_path}")
            except Exception as copy_error:
                logger.error(f"Failed to copy video to output directory: {copy_error}")
                return jsonify({'error': f'Failed to save processed video: {copy_error}'}), 500
            
            return jsonify({
                'success': True,
                'message': 'Video processed successfully',
                'output_file': output_filename
            })
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/output_videos/<filename>')
def download_video(filename):
    return send_from_directory('output_videos', filename)

@app.route('/api/list-videos')
def list_videos():
    """List all processed videos in the output directory."""
    try:
        output_dir = 'output_videos'
        if not os.path.exists(output_dir):
            return jsonify({'videos': []})
        
        videos = []
        for filename in os.listdir(output_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                file_path = os.path.join(output_dir, filename)
                file_size = os.path.getsize(file_path)
                videos.append({
                    'filename': filename,
                    'size': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                })
        
        # Sort by modification time (newest first)
        videos.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x['filename'])), reverse=True)
        
        return jsonify({'videos': videos})
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-video/<filename>')
def delete_video(filename):
    """Delete a processed video file."""
    try:
        output_dir = 'output_videos'
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        os.remove(file_path)
        logger.info(f"Deleted video: {filename}")
        return jsonify({'success': True, 'message': f'Deleted {filename}'})
    except Exception as e:
        logger.error(f"Error deleting video {filename}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-transliteration')
def test_transliteration():
    """Test the transliteration system to ensure it's working correctly."""
    try:
        from src.subtitles import transliterate_arabic_to_latin
        
        test_cases = [
            "شامس",
            "في السماء", 
            "أشكال العديدة",
            "وأحب لنا المساء"
        ]
        
        results = {}
        for arabic_text in test_cases:
            latin_result = transliterate_arabic_to_latin(arabic_text)
            results[arabic_text] = latin_result
        
        return jsonify({
            'success': True,
            'message': 'Transliteration test completed',
            'results': results
        })
    except Exception as e:
        logger.error(f"Transliteration test failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/preload-model/<model_size>')
def preload_model(model_size):
    """Preload a Whisper model to avoid loading delays during video processing."""
    try:
        global whisper_model
        
        if whisper_model is not None:
            current_model = type(whisper_model).__name__
            return jsonify({
                'success': True,
                'message': f'Model already loaded: {current_model}',
                'model': current_model
            })
        
        logger.info(f"Preloading Whisper model: {model_size}")
        whisper_model = load_whisper(model_size)
        logger.info(f"Model preloaded successfully: {type(whisper_model).__name__}")
        
        return jsonify({
            'success': True,
            'message': f'Model {model_size} preloaded successfully',
            'model': type(whisper_model).__name__
        })
        
    except Exception as e:
        logger.error(f"Model preload error: {e}")
        return jsonify({'error': f'Failed to preload model: {str(e)}'}), 500

@app.route('/api/model-status')
def model_status():
    """Get the current status of the loaded Whisper model."""
    try:
        global whisper_model
        
        if whisper_model is None:
            return jsonify({
                'loaded': False,
                'model': None,
                'message': 'No model loaded'
            })
        
        return jsonify({
            'loaded': True,
            'model': type(whisper_model).__name__,
            'message': 'Model ready for processing'
        })
        
    except Exception as e:
        logger.error(f"Model status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/colab-status')
def colab_status():
    """Get detailed information about the Colab environment and system resources."""
    try:
        import psutil
        import torch
        
        # Basic Colab detection
        colab_info = {
            'is_colab': IS_COLAB,
            'colab_gpu': 'COLAB_GPU' in os.environ,
            'colab_tpu': 'COLAB_TPU' in os.environ,
        }
        
        # System resources
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'memory_percent': psutil.virtual_memory().percent,
        }
        
        # PyTorch device info
        torch_info = {
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            torch_info['current_device'] = torch.cuda.current_device()
            torch_info['device_name'] = torch.cuda.get_device_name(0)
            torch_info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        
        # Environment variables
        env_info = {
            'omp_threads': os.environ.get('OMP_NUM_THREADS', 'Not set'),
            'mkl_threads': os.environ.get('MKL_NUM_THREADS', 'Not set'),
            'openblas_threads': os.environ.get('OPENBLAS_NUM_THREADS', 'Not set'),
        }
        
        return jsonify({
            'success': True,
            'colab_info': colab_info,
            'system_info': system_info,
            'torch_info': torch_info,
            'env_info': env_info,
            'message': 'Colab environment status retrieved successfully'
        })
        
    except Exception as e:
        logger.error(f"Colab status error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Colab-specific port selection
    if IS_COLAB:
        port = 12345  # Use port 12345 for Colab
        logger.info(f"🎯 Starting Flask app on Colab port {port}")
    else:
        port = 5001  # Use default port for local development
        logger.info(f"🏠 Starting Flask app on local port {port}")
    
    app.run(host='0.0.0.0', port=port, debug=True)
