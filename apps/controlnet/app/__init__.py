from flask import Flask, request, jsonify, Response
import inference
import json

from .config import Config
from .models.controlnet import init_models

def create_app(config_class=Config):
  app = Flask(__name__)
  app.config.from_object(config_class)

  with app.app_context():
    app.pipe = init_models(app)

  from .routes.sagemaker import sagemaker_bp
  app.register_blueprint(sagemaker_bp)

  @app.route("/ping", methods=['GET'])
  def ping():
    inference.get_model()
    return Response(response=json.dumps({"status": "healthy"}), status=200)

  @app.route('/invocations', methods=['POST'])
  def invoke():
    # inference endpoint required by SM
    try:
      if request.content_type == 'application/json':
        input_data = request.get_json()
      else:
        return Response(
          response=json.dumps({"error": "Unsupported content type"}),
          status=415, mimetype='application/json'
        )
      
      # Extract params
      prompt = input_data.get("prompt")
      if not prompt:
        return Response(
          response=json.dumps({"error": "No prompt provided"}),
          status=400, mimetype='application/json'
        )
      
      model = inference.get_model()
      result = model.generate(
        prompt=prompt,
        negative_prompt=input_data.get("negative_prompt"),
        num_inference_steps=input_data.get("num_inference_steps"),
        controlnet_conditioning_scale=input_data.get("controlnet_conditioning_scale"),
        control_guidance_start=input_data.get("control_guidance_start"),
        control_guidance_end=input_data.get("control_guidance_end"),
        height=input_data.get("height"),
        width=input_data.get("width"),
        sampler=input_data.get("sampler", "dpm++_2m_karras"),
        guidance_scale=input_data.get("guidance_scale", 7.5),
        model=input_data.get("model"),
      )

      return Response(
        response=json.dumps(result),
        status=200, mimetype='application/json'
      )
    
    except Exception as e:
      import traceback
      error_traceback = traceback.format_exc()
      print(f"Error during invocation: {str(e)}")
      print(f"Traceback: {error_traceback}")
      return Response(
        response=json.dumps({
          "error": str(e),
          "traceback": error_traceback
        }),
        status=500, mimetype='application/json'
      )
    
  return app
