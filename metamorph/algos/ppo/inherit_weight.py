import torch

from metamorph.config import cfg


def restore_from_checkpoint(ac, map_location=None):
    load_kwargs = {}
    if map_location is not None:
        load_kwargs["map_location"] = map_location
    try:
        model_p, ob_rms = torch.load(
            cfg.PPO.CHECKPOINT_PATH, weights_only=False, **load_kwargs
        )
    except TypeError:
        # ``weights_only`` is unavailable on older supported PyTorch versions.
        model_p, ob_rms = torch.load(cfg.PPO.CHECKPOINT_PATH, **load_kwargs)

    state_dict_c = ac.state_dict()
    state_dict_p = model_p.state_dict()

    fine_tune_layers = set()
    layer_substrings = cfg.MODEL.FINETUNE.LAYER_SUBSTRING
    for name, param in state_dict_c.items():
        param_p = state_dict_p[name]
        if param_p.shape == param.shape:
            with torch.no_grad():
                param.copy_(param_p)
        else:
            raise ValueError(
                "Checkpoint path is invalid as there is shape mismatch"
            )
        if any(name_substr in name for name_substr in layer_substrings):
            fine_tune_layers.add(name)

    if not cfg.MODEL.FINETUNE.FULL_MODEL:
        for name, param in ac.named_parameters():
            if name not in fine_tune_layers:
                param.requires_grad = False

    return ob_rms
