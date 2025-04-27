from marshmallow import Schema, fields, validate, validates, ValidationError

class GenerateImageSchema(Schema):
    """Schema for validating image generation requests"""
    
    default_negative_prompt = "ugly, disfigured, low quality, blurry, nsfw"
    default_num_inference_steps = 20 
    default_controlnet_conditioning_scale = [1.25, 0.1]
    default_control_guidance_start = [0, 0.1]
    default_control_guidance_end = [1, 1]
    default_size = 512

    prompt = fields.String(required=True)
    input_image = fields.String(required=True) 
    negative_prompt = fields.String(missing=default_negative_prompt)
    num_inference_steps = fields.Integer(missing=default_num_inference_steps)
    controlnet_conditioning_scale = fields.List(fields.Float, missing=default_controlnet_conditioning_scale)
    control_guidance_start = fields.List(fields.Float, missing=default_control_guidance_start)
    control_guidance_end = fields.List(fields.Float, missing=default_control_guidance_end)
    height = fields.Integer(missing=default_size)
    width = fields.Integer(missing=default_size)
    device = fields.String(missing=None)

    @validates('controlnet_conditioning_scale')
    def validate_controlnet_conditioning_scale(self, value):
        if len(value) != 2:
            raise ValidationError('controlnet_conditioning_scale must be a list of two numbers.')

    @validates('control_guidance_start')
    def validate_control_guidance_start(self, value):
        if len(value) != 2:
            raise ValidationError('control_guidance_start must be a list of two numbers.')

    @validates('control_guidance_end')
    def validate_control_guidance_end(self, value):
        if len(value) != 2:
            raise ValidationError('control_guidance_end must be a list of two numbers.')
            
    @validates('height')
    def validate_height(self, value):
        if value % 8 != 0:
            raise ValidationError('Height must be divisible by 8.')
        if value > 1024:
            raise ValidationError('Height cannot exceed 1024 pixels.')
        
    @validates('width')
    def validate_width(self, value):
        if value % 8 != 0:
            raise ValidationError('Width must be divisible by 8.')
        if value > 1024:
            raise ValidationError('Width cannot exceed 1024 pixels.')
            
    @validates('device')
    def validate_device(self, value):
        if value and value not in ['auto', 'cpu', 'mps', 'cuda', None]:
            raise ValidationError('Device must be one of: auto, cpu, mps, cuda')

class SageMakerRequestSchema(Schema):
    """Schema for validating SageMaker async generation requests"""
    prompt = fields.String(required=True, validate=validate.Length(min=1, max=1000))
    negative_prompt = fields.String(validate=validate.Length(max=1000))
    num_inference_steps = fields.Integer(validate=validate.Range(min=1, max=100))
    controlnet_conditioning_scale = fields.List(fields.Float())
    control_guidance_start = fields.List(fields.Float())
    control_guidance_end = fields.List(fields.Float())
    height = fields.Integer(validate=validate.OneOf([512, 768, 1024]))
    width = fields.Integer(validate=validate.OneOf([512, 768, 1024]))