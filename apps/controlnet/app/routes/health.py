from flask import Blueprint, jsonify, current_app
import torch
from ..utils.logging import get_logger
import platform
import os

logger = get_logger(__name__)
health_bp = Blueprint('health', __name__)

@health_bp.route('/ping', methods=['GET'])
def ping():
  return "OK", 200

@health_bp.route('/health', methods=['GET'])
def health():
  try:
      model_loaded = current_app.pipe is not None
      
      device = current_app.config['DEVICE']
      
      memory_info = {}
      if device == "cuda" and torch.cuda.is_available():
          memory_info = {
              "total_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024),
              "allocated_memory_mb": round(torch.cuda.memory_allocated() / 1024 / 1024),
              "reserved_memory_mb": round(torch.cuda.memory_reserved() / 1024 / 1024),
          }
      
      system_info = {
          "platform": platform.platform(),
          "python_version": platform.python_version(),
          "pytorch_version": torch.__version__,
      }
      
      results_dir = current_app.config['RESULTS_DIR']
      results_dir_exists = os.path.isdir(results_dir)
      
      return jsonify({
          "status": "healthy",
          "model_loaded": model_loaded,
          "device": device,
          "cuda_available": torch.cuda.is_available(),
          "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
          "memory_info": memory_info,
          "system_info": system_info,
          "results_directory": {
              "path": results_dir,
              "exists": results_dir_exists
          }
      }), 200
  except Exception as e:
      logger.error(f"Health check failed: {str(e)}")
      return jsonify({
          "status": "unhealthy",
          "error": str(e)
      }), 500