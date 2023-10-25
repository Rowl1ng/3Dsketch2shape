from models.styleflow.flow import build_model

def get_flow(style_dim=48, width=512, depth=4, condition_size=2):
    return build_model(style_dim, (width,) * depth, condition_size, 1, True)
