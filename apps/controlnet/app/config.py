import os

class Config:
  DEBUG = os.environ.get("DEBUG", 'False') == 'True'

  LOG_LEVEL = os.environ.get("LOG_LEVEL", 'INFO')

  MODEL = os.environ.get("MODEL", "Lykon/DreamShaper")
  CONTROLNET_MODEL = os.environ.get('CONTROLNET_MODEL', "monster-labs/control_v1p_sd15_qrcode_monster")
  CONTROLNET_TWO_MODEL = os.environ.get('CONTROLNET_TWO_MODEL', "latentcat/control_v1p_sd15_brightness")

  DEVICE = "cpu"

  NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", 30))

  RESULTS_DIR = os.environ.get('RESULTS_DIR', './results')