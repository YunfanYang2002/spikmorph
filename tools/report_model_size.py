"""Report the Actor-Critic parameter budget without constructing an environment.

The calculation mirrors the linear layers and normalization parameters in
metamorph.algos.ppo.model. It is useful in CI and on machines without MuJoCo or
SpikingJelly. The default limb observation width is derived from the currently
selected proprioceptive features (30 limb features plus two 11-feature joints).
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metamorph.config import cfg


FEATURE_WIDTHS = {
    "body_xpos": 3,
    "body_xvelp": 3,
    "body_xvelr": 3,
    "body_xquat": 4,
    "body_pos": 3,
    "body_ipos": 3,
    "body_iquat": 4,
    "geom_quat": 4,
    "body_mass": 1,
    "body_shape": 2,
    "qpos": 1,
    "qvel": 1,
    "jnt_pos": 3,
    "joint_range": 2,
    "joint_axis": 3,
    "gear": 1,
}

JOINT_FEATURES = {
    "qpos", "qvel", "jnt_pos", "joint_range", "joint_axis", "gear"
}


def linear_params(in_features, out_features):
    return in_features * out_features + out_features


def infer_limb_obs_size(obs_types):
    unknown = sorted(set(obs_types) - set(FEATURE_WIDTHS))
    if unknown:
        raise ValueError(
            "Unknown proprioceptive feature widths: {}. Pass --limb-obs-size."
            .format(", ".join(unknown))
        )
    limb_width = sum(
        FEATURE_WIDTHS[name] for name in obs_types if name not in JOINT_FEATURES
    )
    joint_width = sum(
        FEATURE_WIDTHS[name] for name in obs_types if name in JOINT_FEATURES
    )
    return limb_width + 2 * joint_width


def mlp_params(dims):
    return sum(linear_params(dim_in, dim_out) for dim_in, dim_out in zip(dims, dims[1:]))


def encoder_layer_params(d_model, dim_feedforward):
    attention = 4 * linear_params(d_model, d_model)
    ffn = linear_params(d_model, dim_feedforward) + linear_params(
        dim_feedforward, d_model
    )
    layer_norms = 4 * d_model
    return {"attention": attention, "ffn": ffn, "layer_norms": layer_norms}


def model_parameter_budget(limb_obs_size, hfield_obs_dim=None):
    model_cfg = cfg.MODEL
    transformer_cfg = model_cfg.TRANSFORMER
    d_model = model_cfg.LIMB_EMBED_SIZE
    layer = encoder_layer_params(d_model, transformer_cfg.DIM_FEEDFORWARD)
    encoder = sum(layer.values()) * transformer_cfg.NLAYERS

    embedding = linear_params(limb_obs_size, d_model)
    position = (
        model_cfg.MAX_LIMBS * d_model
        if transformer_cfg.POS_EMBEDDING == "learnt"
        else 0
    )

    task_encoder = 0
    task_feature_dim = 0
    if "hfield" in cfg.ENV.KEYS_TO_KEEP:
        if hfield_obs_dim is None:
            raise ValueError(
                "The selected config uses hfield observations; pass --hfield-obs-dim."
            )
        external_dims = list(transformer_cfg.EXT_HIDDEN_DIMS)
        if not external_dims:
            raise ValueError("EXT_HIDDEN_DIMS must be non-empty when hfield is enabled.")
        task_encoder = mlp_params([hfield_obs_dim] + external_dims)
        if transformer_cfg.EXT_MIX == "late":
            task_feature_dim = external_dims[-1]

    decoder_input = d_model + task_feature_dim
    critic_decoder = mlp_params(
        [decoder_input] + list(transformer_cfg.DECODER_DIMS) + [1]
    )
    actor_decoder = mlp_params(
        [decoder_input] + list(transformer_cfg.DECODER_DIMS) + [2]
    )

    shared_per_network = embedding + position + encoder + task_encoder
    critic = shared_per_network + critic_decoder
    actor = shared_per_network + actor_decoder
    action_std = model_cfg.MAX_LIMBS * 2
    total = critic + actor + action_std
    trainable = total if not model_cfg.ACTION_STD_FIXED else total - action_std

    return {
        "limb_obs_size": limb_obs_size,
        "attention_per_layer": layer["attention"],
        "ffn_per_layer": layer["ffn"],
        "norm_per_layer": layer["layer_norms"],
        "encoder_per_network": encoder,
        "critic_network": critic,
        "actor_network": actor,
        "fixed_action_std": action_std if model_cfg.ACTION_STD_FIXED else 0,
        "trainable": trainable,
        "total": total,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="YAML configuration file")
    parser.add_argument("--limb-obs-size", type=int)
    parser.add_argument("--hfield-obs-dim", type=int)
    parser.add_argument(
        "--expect-max-params",
        type=int,
        help="Exit with an error when total parameters exceed this budget",
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Additional yacs configuration overrides"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    limb_obs_size = args.limb_obs_size
    if limb_obs_size is None:
        limb_obs_size = infer_limb_obs_size(cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES)

    budget = model_parameter_budget(limb_obs_size, args.hfield_obs_dim)
    baseline = 3314579
    reduction = 100.0 * (1.0 - budget["total"] / baseline)

    print("Encoder type: {}".format(cfg.MODEL.ENCODER_TYPE))
    print(
        "Shape: D={}, FFN={}, layers={}, limb_obs={}".format(
            cfg.MODEL.LIMB_EMBED_SIZE,
            cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
            cfg.MODEL.TRANSFORMER.NLAYERS,
            limb_obs_size,
        )
    )
    for name in [
        "attention_per_layer",
        "ffn_per_layer",
        "norm_per_layer",
        "encoder_per_network",
        "critic_network",
        "actor_network",
        "trainable",
        "total",
    ]:
        print("{}: {:,}".format(name, budget[name]))
    print("Reduction vs. 3,314,579 baseline: {:.2f}%".format(reduction))

    if args.expect_max_params is not None and budget["total"] > args.expect_max_params:
        raise SystemExit(
            "Parameter budget exceeded: {:,} > {:,}".format(
                budget["total"], args.expect_max_params
            )
        )


if __name__ == "__main__":
    main()
