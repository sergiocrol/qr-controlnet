import os
import json
import flask
import inference

app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
  # This health endpoint is required by SM
  inference.get_model()
  return flask.Response(response=json.dumps({"status": "healthy"}), status=200)

@app.route('/invocations', methods=['POST'])
def invoke():
  # inference endpoint required by SM
  try:
    if flask.request.content_type == 'application/json':
      input_data = flask.request.get_json()
    else:
      return flask.Response(
        response=json.dumps({"error": "Unsupported content type"}),
        status=415, mimetype='application/json'
      )
    
    # Extract params
    prompt = input_data.get("prompt")
    if not prompt:
      return flask.Response(
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
    )

    return flask.Response(
      response=json.dumps(result),
      status=200, mimetype='application/json'
    )
  
  except Exception as e:
    import traceback
    error_traceback = traceback.format_exc()
    print(f"Error during invocation: {str(e)}")
    print(f"Traceback: {error_traceback}")
    return flask.Response(
      response=json.dumps({
        "error": str(e),
        "traceback": error_traceback
      }),
      status=500, mimetype='application/json'
    )
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)