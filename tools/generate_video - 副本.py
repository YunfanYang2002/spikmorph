import argparse
import math
import os
import subprocess
import sys
import tempfile
import warnings

import imageio
import numpy as np
import torch
from PIL import Image

# Monkey patch for numpy >= 1.20 compatibility without triggering FutureWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in cast",
    )
    np_dict = np.__dict__
    if np_dict.get("bool", None) is None:
        np.bool = bool
    if np_dict.get("int", None) is None:
        np.int = int
    if np_dict.get("float", None) is None:
        np.float = float
    if np_dict.get("complex", None) is None:
        np.complex = complex
    if np_dict.get("object", None) is None:
        np.object = object
    if np_dict.get("str", None) is None:
        np.str = str

from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import get_ob_rms, make_vec_envs, set_ob_rms
from metamorph.config import cfg
from metamorph.utils import file as fu
from metamorph.utils import sample as su


def log_info(message):
    print("[generate_video] {}".format(message))


def parse_csv_arg(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_frame(frame):
    if frame is None:
        raise ValueError("Render returned None for a video frame.")

    if isinstance(frame, (list, tuple)):
        if len(frame) == 0:
            raise ValueError("Render returned an empty frame list.")
        frame = frame[0]

    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            "Invalid frame shape for video: {}. Expected HxWx3.".format(frame.shape)
        )

    frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
    if not np.issubdtype(frame.dtype, np.integer):
        frame = np.rint(frame)
    return np.clip(frame, 0, 255).astype(np.uint8)


def resize_frame(frame, width, height):
    image = Image.fromarray(frame)
    image = image.resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def save_gif(frames, gif_path, fps):
    if not frames:
        raise RuntimeError("Cannot save GIF with no frames.")
    duration = 1.0 / float(fps) if fps else 0.1
    imageio.mimsave(gif_path, frames, format="GIF", duration=duration)


def rewrite_video_for_compatibility(video_path):
    try:
        import imageio_ffmpeg
    except ImportError:
        return video_path

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    temp_output = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="generate_video_compat_", suffix=".mp4", delete=False
        ) as tmp_file:
            temp_output = tmp_file.name

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            video_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            temp_output,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg rewrite failed")

        os.replace(temp_output, video_path)
        temp_output = None
        return video_path
    finally:
        if temp_output is not None and os.path.exists(temp_output):
            os.remove(temp_output)


class StreamingVideoWriter:
    def __init__(self, video_path, fps, frame_shape):
        self.video_path = video_path
        self.fps = fps
        self.frame_shape = frame_shape
        self.backend = None
        self.writer = None

        height, width = frame_shape[:2]
        try:
            import imageio_ffmpeg

            self.backend = "imageio_ffmpeg"
            self.writer = imageio_ffmpeg.write_frames(
                video_path,
                (width, height),
                pix_fmt_in="rgb24",
                fps=fps,
                codec="libx264",
                quality=5,
            )
            self.writer.send(None)
        except ImportError:
            self.backend = "imageio"
            self.writer = imageio.get_writer(video_path, fps=fps)

    def append(self, frame):
        if frame.shape != self.frame_shape:
            raise ValueError(
                "Video frame shape changed from {} to {}.".format(
                    self.frame_shape, frame.shape
                )
            )

        if self.backend == "imageio_ffmpeg":
            self.writer.send(frame)
        else:
            self.writer.append_data(frame)

    def close(self):
        if self.writer is None:
            return

        if self.backend == "imageio_ffmpeg":
            self.writer.close()
        else:
            self.writer.close()
        self.writer = None

        if not os.path.exists(self.video_path) or os.path.getsize(self.video_path) == 0:
            raise RuntimeError("Failed to create video file: {}".format(self.video_path))

        rewrite_video_for_compatibility(self.video_path)


def reset_cfg_from(snapshot):
    cfg.clear()
    cfg.update(snapshot.clone())


def calculate_max_iters():
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS)
        // cfg.PPO.TIMESTEPS
        // cfg.PPO.NUM_ENVS
    )


def maybe_infer_walkers():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    if len(cfg.ENV.WALKERS):
        return

    xml_dir = os.path.join(cfg.ENV.WALKER_DIR, "xml")
    if not os.path.isdir(xml_dir):
        raise FileNotFoundError(
            "Walker xml directory not found: {}. "
            "Please set --walker-dir or cfg.ENV.WALKER_DIR correctly.".format(xml_dir)
        )

    cfg.ENV.WALKERS = [
        xml_file.split(".")[0]
        for xml_file in os.listdir(xml_dir)
        if xml_file.endswith(".xml")
    ]

    if not cfg.ENV.WALKERS:
        raise ValueError("No walker xml files found in {}.".format(xml_dir))


def calculate_max_limbs_joints():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    num_joints, num_limbs = [], []
    for agent in cfg.ENV.WALKERS:
        metadata_path = os.path.join(
            cfg.ENV.WALKER_DIR, "metadata", "{}.json".format(agent)
        )
        metadata = fu.load_json(metadata_path)
        num_joints.append(metadata["dof"])
        num_limbs.append(metadata["num_limbs"] + 1)

    cfg.MODEL.MAX_JOINTS = max(num_joints) + 1
    cfg.MODEL.MAX_LIMBS = max(num_limbs) + 1


def set_cfg_options():
    calculate_max_iters()
    maybe_infer_walkers()
    calculate_max_limbs_joints()


def resolve_checkpoint_path():
    checkpoint_path = cfg.PPO.CHECKPOINT_PATH or os.path.join(
        cfg.OUT_DIR, cfg.ENV_NAME + ".pt"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Checkpoint file not found: {}. Please set PPO.CHECKPOINT_PATH.".format(
                checkpoint_path
            )
        )
    return checkpoint_path


def maybe_strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    if all(isinstance(key, str) and key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def extract_checkpoint_payload(checkpoint):
    state_dict = None
    ob_rms = None

    if isinstance(checkpoint, (list, tuple)):
        if len(checkpoint) >= 1:
            first = checkpoint[0]
            state_dict = first.state_dict() if hasattr(first, "state_dict") else first
        if len(checkpoint) >= 2:
            ob_rms = checkpoint[1]
    elif isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "actor_critic" in checkpoint:
            state_dict = checkpoint["actor_critic"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        for key in ("ob_rms", "obs_rms", "observation_rms"):
            if key in checkpoint:
                ob_rms = checkpoint[key]
                break
    else:
        state_dict = checkpoint

    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    return maybe_strip_module_prefix(state_dict), ob_rms


def load_checkpoint_into_trainer(trainer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(cfg.DEVICE))
    state_dict, ob_rms = extract_checkpoint_payload(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Unsupported checkpoint payload type: {}.".format(type(state_dict))
        )

    current_state = trainer.actor_critic.state_dict()
    compatible_state = {}
    mismatched_keys = []
    for key, value in state_dict.items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible_state[key] = value
        elif key in current_state:
            mismatched_keys.append(
                "{}: checkpoint {} vs current {}".format(
                    key, tuple(value.shape), tuple(current_state[key].shape)
                )
            )

    if not compatible_state:
        raise RuntimeError(
            "Checkpoint is incompatible with the current scene/model config. "
            "This usually happens when the checkpoint was trained on a different "
            "observation structure, for example `configs/ft.yaml` (flat floor) "
            "versus `configs/obstacle.yaml` or `configs/csr.yaml` (which add "
            "`Terrain`/`hfield`). Checkpoint: {}. Example mismatches: {}".format(
                checkpoint_path,
                ", ".join(mismatched_keys[:5]) if mismatched_keys else "none"
            )
        )

    trainer.actor_critic.load_state_dict(compatible_state, strict=False)
    trainer.actor_critic.eval()

    if ob_rms is not None:
        set_ob_rms(trainer.envs, ob_rms)
    return ob_rms


def render_rgb_frame(env):
    try:
        return normalize_frame(env.render(mode="rgb_array"))
    except TypeError:
        return normalize_frame(env.render())


def build_scene_specs(args, base_task_name):
    if args.scene_cfgs:
        scene_paths = parse_csv_arg(args.scene_cfgs)
        specs = []
        for scene_path in scene_paths:
            scene_name = os.path.splitext(os.path.basename(scene_path))[0]
            specs.append({"name": scene_name, "path": scene_path})
        return specs

    if args.showcase:
        return [{"name": base_task_name, "path": None}]

    return [{"name": base_task_name, "path": None}]


def select_walkers(args):
    if args.walkers:
        return parse_csv_arg(args.walkers)
    if args.walker_name:
        return [args.walker_name]

    if len(cfg.ENV.WALKERS) == 0:
        maybe_infer_walkers()

    if args.showcase:
        return cfg.ENV.WALKERS[: min(5, len(cfg.ENV.WALKERS))]
    return [cfg.ENV.WALKERS[0]]


def apply_scene_configuration(base_snapshot, scene_path, walker_name):
    reset_cfg_from(base_snapshot)
    if scene_path is not None:
        cfg.merge_from_file(scene_path)

    cfg.ENV.WALKERS = [walker_name]
    cfg.PPO.NUM_ENVS = 1
    cfg.VECENV.TYPE = "DummyVecEnv"
    set_cfg_options()


def get_video_dir():
    video_dir = os.path.join(cfg.OUT_DIR, "videos")
    os.makedirs(video_dir, exist_ok=True)
    return video_dir


def get_clip_output_path(video_dir, walker_name, scene_name):
    return os.path.join(video_dir, "{}_{}_{}.mp4".format(cfg.ENV_NAME, walker_name, scene_name))


def get_sample_indices(total_frames, max_frames):
    if max_frames <= 0 or max_frames >= total_frames:
        return list(range(total_frames))
    indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int).tolist()
    return sorted(set(indices))


def collect_clip(scene_spec, walker_name, base_snapshot, args):
    apply_scene_configuration(base_snapshot, scene_spec["path"], walker_name)
    checkpoint_path = resolve_checkpoint_path()
    su.set_seed(cfg.RNG_SEED)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.set_num_threads(1)

    trainer = None
    video_env = None
    writer = None
    original_checkpoint_path = cfg.PPO.CHECKPOINT_PATH
    try:
        cfg.PPO.CHECKPOINT_PATH = ""
        trainer = PPO(print_model=False)
        cfg.PPO.CHECKPOINT_PATH = original_checkpoint_path
        ob_rms = load_checkpoint_into_trainer(trainer, checkpoint_path)

        video_env = make_vec_envs(training=False, norm_rew=False, num_env=1)
        if ob_rms is None:
            ob_rms = get_ob_rms(trainer.envs)
        if ob_rms is not None:
            set_ob_rms(video_env, ob_rms)

        video_dir = get_video_dir()
        video_path = get_clip_output_path(video_dir, walker_name, scene_spec["name"])
        total_frames = cfg.PPO.VIDEO_LENGTH + 1
        sample_indices = get_sample_indices(total_frames, args.gif_max_frames)
        sample_lookup = set(sample_indices)
        sampled_frames = []

        obs = video_env.reset()
        first_frame = render_rgb_frame(video_env)
        writer = StreamingVideoWriter(video_path, cfg.VIDEO.FPS, first_frame.shape)
        writer.append(first_frame)
        if 0 in sample_lookup:
            sampled_frames.append(
                resize_frame(first_frame, args.tile_width, args.tile_height)
            )

        with torch.no_grad():
            for step_idx in range(cfg.PPO.VIDEO_LENGTH):
                _, act, _ = trainer.agent.act(obs)
                obs, _, _, _ = video_env.step(act)
                frame = render_rgb_frame(video_env)
                writer.append(frame)
                frame_idx = step_idx + 1
                if frame_idx in sample_lookup:
                    sampled_frames.append(
                        resize_frame(frame, args.tile_width, args.tile_height)
                    )

        writer.close()
        writer = None

        return {
            "walker": walker_name,
            "scene": scene_spec["name"],
            "video_path": video_path,
            "gif_frames": sampled_frames,
        }
    finally:
        cfg.PPO.CHECKPOINT_PATH = original_checkpoint_path
        if writer is not None:
            writer.close()
        if video_env is not None:
            video_env.close()
        if trainer is not None and getattr(trainer, "envs", None) is not None:
            trainer.envs.close()


def compose_montage_frames(clips, cols, tile_width, tile_height, padding):
    if not clips:
        raise RuntimeError("No clips available for montage composition.")

    frame_count = min(len(clip["gif_frames"]) for clip in clips)
    if frame_count == 0:
        raise RuntimeError("Montage clips do not contain any GIF frames.")

    rows = int(math.ceil(len(clips) / float(cols)))
    canvas_width = cols * tile_width + (cols - 1) * padding
    canvas_height = rows * tile_height + (rows - 1) * padding
    montage_frames = []

    for frame_idx in range(frame_count):
        canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
        for clip_idx, clip in enumerate(clips):
            row = clip_idx // cols
            col = clip_idx % cols
            y0 = row * (tile_height + padding)
            x0 = col * (tile_width + padding)
            canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = clip["gif_frames"][
                frame_idx
            ]
        montage_frames.append(canvas)

    return montage_frames


def save_montage_outputs(clips, args):
    montage_frames = compose_montage_frames(
        clips, args.grid_cols, args.tile_width, args.tile_height, args.grid_padding
    )
    video_dir = get_video_dir()
    output_name = args.output_name or "metamorph_showcase"
    gif_path = os.path.join(video_dir, output_name + ".gif")
    mp4_path = os.path.join(video_dir, output_name + ".mp4")

    save_gif(montage_frames, gif_path, args.gif_fps or cfg.VIDEO.FPS)

    writer = StreamingVideoWriter(mp4_path, args.gif_fps or cfg.VIDEO.FPS, montage_frames[0].shape)
    try:
        for frame in montage_frames:
            writer.append(frame)
    finally:
        writer.close()

    return {"gif": gif_path, "mp4": mp4_path}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MetaMorph rollout videos and teaser-style showcases."
    )
    parser.add_argument("--cfg", dest="cfg_file", required=True, type=str, help="Config file")
    parser.add_argument(
        "--walker-dir",
        dest="walker_dir",
        default=None,
        type=str,
        help="Override cfg.ENV.WALKER_DIR",
    )
    parser.add_argument(
        "--walker",
        dest="walker_name",
        default=None,
        type=str,
        help="Single walker name without extension",
    )
    parser.add_argument(
        "--walkers",
        dest="walkers",
        default=None,
        type=str,
        help="Comma-separated walker names for batch/showcase generation",
    )
    parser.add_argument(
        "--showcase",
        action="store_true",
        help="Generate a teaser-style grid using multiple walkers and scenes",
    )
    parser.add_argument(
        "--scene-cfgs",
        dest="scene_cfgs",
        default=None,
        type=str,
        help="Comma-separated scene config files, e.g. configs/obstacle.yaml,configs/csr.yaml",
    )
    parser.add_argument(
        "--output-name",
        dest="output_name",
        default=None,
        type=str,
        help="Base output name for showcase montage files",
    )
    parser.add_argument("--grid-cols", dest="grid_cols", default=5, type=int)
    parser.add_argument("--grid-padding", dest="grid_padding", default=4, type=int)
    parser.add_argument("--tile-width", dest="tile_width", default=384, type=int)
    parser.add_argument("--tile-height", dest="tile_height", default=384, type=int)
    parser.add_argument(
        "--gif-max-frames",
        dest="gif_max_frames",
        default=101,
        type=int,
        help="Maximum sampled frames kept for montage gif/mp4 composition",
    )
    parser.add_argument(
        "--gif-fps",
        dest="gif_fps",
        default=12,
        type=int,
        help="FPS for teaser-style gif/mp4 montage",
    )
    parser.add_argument(
        "opts",
        help="Additional config overrides",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        cfg.merge_from_file(args.cfg_file)
        cfg.merge_from_list(args.opts)

        if args.walker_dir is not None:
            cfg.ENV.WALKER_DIR = args.walker_dir

        set_cfg_options()
        base_snapshot = cfg.clone()
        walkers = select_walkers(args)
        if not walkers:
            raise ValueError("No walkers available for video generation.")

        scene_specs = build_scene_specs(args, cfg.ENV.TASK)
        total_clips = len(walkers) * len(scene_specs)
        log_info(
            "Starting video generation for {} clip(s): {} walker(s) across {} scene(s).".format(
                total_clips, len(walkers), len(scene_specs)
            )
        )

        clips = []
        for scene_spec in scene_specs:
            for walker_name in walkers:
                clips.append(collect_clip(scene_spec, walker_name, base_snapshot, args))

        if len(clips) > 1 or args.showcase or args.scene_cfgs:
            outputs = save_montage_outputs(clips, args)
            log_info(
                "Finished video generation. Saved {} individual clip(s), showcase GIF at {}, and showcase MP4 at {}.".format(
                    len(clips), outputs["gif"], outputs["mp4"]
                )
            )
        else:
            log_info(
                "Finished video generation. Saved video to {}.".format(
                    clips[0]["video_path"]
                )
            )
    except Exception as exc:
        print("[generate_video] ERROR: {}".format(exc))
        raise


if __name__ == "__main__":
    main()
