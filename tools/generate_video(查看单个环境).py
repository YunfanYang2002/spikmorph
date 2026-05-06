import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import warnings

# Monkey patch for numpy >= 1.20 compatibility without triggering FutureWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    np_dict = np.__dict__
    if np_dict.get('bool', None) is None:
        np.bool = bool
    if np_dict.get('int', None) is None:
        np.int = int
    if np_dict.get('float', None) is None:
        np.float = float
    if np_dict.get('complex', None) is None:
        np.complex = complex
    if np_dict.get('object', None) is None:
        np.object = object
    if np_dict.get('str', None) is None:
        np.str = str

import imageio
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import get_ob_rms, make_vec_envs, set_ob_rms
from metamorph.config import cfg
from metamorph.config import dump_cfg
from metamorph.envs.vec_env.vec_video_recorder import VecVideoRecorder
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu


def log_info(message):
    print("[generate_video] {}".format(message))


def debug_frame(frame, prefix="frame"):
    if frame is None:
        log_info(f"{prefix}: None")
        return
    if isinstance(frame, (list, tuple)):
        log_info(f"{prefix}: list/tuple length={len(frame)} type={type(frame)}")
        if len(frame) > 0:
            frame = frame[0]
    arr = np.asarray(frame)
    log_info(
        f"{prefix}: type={type(frame)}, shape={arr.shape}, dtype={arr.dtype}, "
        f"min={np.nanmin(arr) if arr.size else 'empty'}, max={np.nanmax(arr) if arr.size else 'empty'}"
    )
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        log_info(f"{prefix}: pixel sample {arr.flat[:10].tolist()}")


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
            f"Invalid frame shape for video: {frame.shape}. Expected HxWx3."
        )
    frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
    if not np.issubdtype(frame.dtype, np.integer):
        frame = np.rint(frame)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def write_debug_image(image_path, frame):
    try:
        imageio.imwrite(image_path, normalize_frame(frame))
        log_info("Wrote debug image: {}".format(image_path))
    except Exception as exc:
        log_info("Failed to write debug image {}: {}".format(image_path, exc))


def write_debug_gif(gif_path, frames, fps):
    try:
        normalized_frames = [normalize_frame(frame) for frame in frames]
        duration_ms = max(1, int(round(1000.0 / fps))) if fps else 100
        imageio.mimsave(gif_path, normalized_frames, format="GIF", duration=duration_ms / 1000.0)
        log_info("Wrote debug GIF: {}".format(gif_path))
    except Exception as exc:
        log_info("Failed to write debug GIF {}: {}".format(gif_path, exc))


def cleanup_meta_file(video_path):
    meta_path = os.path.splitext(video_path)[0] + ".meta.json"
    if os.path.exists(meta_path):
        try:
            os.remove(meta_path)
            log_info("Removed monitor metadata file: {}".format(meta_path))
        except Exception as exc:
            log_info("Failed to remove metadata file {}: {}".format(meta_path, exc))


def get_expected_video_path(video_dir, env_name, walker_name):
    return os.path.join(video_dir, "{}_{}_video.mp4".format(env_name, walker_name))


def rewrite_video_for_compatibility(video_path):
    try:
        import imageio_ffmpeg
    except ImportError:
        log_info(
            "imageio_ffmpeg not available; skipping compatibility rewrite for {}".format(
                video_path
            )
        )
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
        log_info(
            "Rewriting video for compatibility with ffmpeg: {}".format(
                " ".join(cmd[:-1] + ["<temp_output>"])
            )
        )
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if result.stdout.strip():
            log_info("ffmpeg rewrite stdout: {}".format(result.stdout.strip()))
        if result.stderr.strip():
            log_info("ffmpeg rewrite stderr: {}".format(result.stderr.strip()))

        if result.returncode != 0:
            log_info(
                "ffmpeg compatibility rewrite failed with return code {}; keeping original file.".format(
                    result.returncode
                )
            )
            return video_path

        os.replace(temp_output, video_path)
        temp_output = None
        log_info("Compatibility rewrite completed and replaced original video.")
        return video_path
    except Exception as exc:
        log_info(
            "Compatibility rewrite failed for {}: {}. Keeping original file.".format(
                video_path, exc
            )
        )
        return video_path
    finally:
        if temp_output is not None and os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except Exception:
                pass


def save_video_from_frames(frames, video_path, fps):
    if len(frames) == 0:
        raise RuntimeError("No frames captured for video.")

    expected_duration = len(frames) / float(fps) if fps else 0.0
    log_info(
        "Saving {} frames to {} at {} FPS (expected duration: {:.2f}s)".format(
            len(frames), video_path, fps, expected_duration
        )
    )
    for idx, frame in enumerate(frames[:5]):
        debug_frame(frame, prefix=f"initial frame[{idx}]")

    frames = [normalize_frame(frame) for frame in frames]
    for idx, frame in enumerate(frames[:5]):
        debug_frame(frame, prefix=f"normalized frame[{idx}]")

    unique_shapes = sorted({frame.shape for frame in frames})
    unique_dtypes = sorted({str(frame.dtype) for frame in frames})
    log_info("Normalized frame shapes: {}".format(unique_shapes))
    log_info("Normalized frame dtypes: {}".format(unique_dtypes))
    if len(unique_shapes) != 1:
        raise ValueError(
            "Captured frames have inconsistent shapes: {}. "
            "This is a common mujoco-py rendering failure mode and will produce invalid video.".format(
                unique_shapes
            )
        )

    if os.path.exists(video_path):
        existing_size = os.path.getsize(video_path)
        log_info(
            "Output path already exists and will be overwritten. Existing size: {} bytes".format(
                existing_size
            )
        )

    try:
        import imageio_ffmpeg
    except ImportError:
        imageio_ffmpeg = None

    if imageio_ffmpeg is not None:
        log_info("Writing video using imageio_ffmpeg fallback writer.")
        height, width = frames[0].shape[:2]
        writer = imageio_ffmpeg.write_frames(
            video_path,
            (width, height),
            pix_fmt_in="rgb24",
            fps=fps,
            codec="libx264",
            quality=5,
        )
        try:
            writer.send(None)
            for idx, frame in enumerate(frames):
                if idx < 3 or idx == len(frames) - 1 or (idx + 1) % 100 == 0:
                    log_info(
                        "Sending frame {} / {} to ffmpeg writer".format(
                            idx + 1, len(frames)
                        )
                    )
                writer.send(frame)
        finally:
            writer.close()
        if verify_written_video(video_path, len(frames), fps, writer_name="imageio_ffmpeg"):
            video_path = rewrite_video_for_compatibility(video_path)
            verify_written_video(
                video_path, len(frames), fps, writer_name="imageio_ffmpeg-compatible"
            )
            return
        log_info("ffmpeg writer output did not validate cleanly; trying imageio writer fallback.")

    log_info("imageio_ffmpeg not available; falling back to imageio writer.")
    with imageio.get_writer(video_path, fps=fps) as writer:
        for idx, frame in enumerate(frames):
            if idx < 3 or idx == len(frames) - 1 or (idx + 1) % 100 == 0:
                log_info(
                    "Appending frame {} / {} to imageio writer".format(
                        idx + 1, len(frames)
                    )
                )
            writer.append_data(frame)
    if verify_written_video(video_path, len(frames), fps, writer_name="imageio"):
        video_path = rewrite_video_for_compatibility(video_path)
        verify_written_video(
            video_path, len(frames), fps, writer_name="imageio-compatible"
        )
        return
    raise RuntimeError(
        "Video file was written but validation still failed for {}. "
        "This often points to a mujoco-py rendering or MP4 container finalization issue.".format(
            video_path
        )
    )


def verify_written_video(video_path, expected_frame_count, fps, writer_name):
    if not os.path.exists(video_path):
        raise RuntimeError(
            "Video writer ({}) completed without creating file: {}".format(
                writer_name, video_path
            )
        )

    file_size = os.path.getsize(video_path)
    log_info(
        "Video writer ({}) finished. Output size: {} bytes".format(
            writer_name, file_size
        )
    )

    if file_size == 0:
        raise RuntimeError("Generated video file is empty: {}".format(video_path))

    reader_ok = False
    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        log_info("imageio reader metadata: {}".format(meta))
        try:
            first_frame = reader.get_data(0)
            debug_frame(first_frame, prefix="decoded first frame")
            reader_ok = True
        finally:
            reader.close()
    except Exception as exc:
        log_info("Failed to read generated video back with imageio: {}".format(exc))

    ffmpeg_ok = False
    try:
        import imageio_ffmpeg

        ffmpeg_meta = imageio_ffmpeg.count_frames_and_secs(video_path)
        log_info(
            "ffmpeg probe result: frame_count={}, duration={:.4f}s, expected_frames={}, expected_duration={:.4f}s".format(
                ffmpeg_meta[0],
                ffmpeg_meta[1],
                expected_frame_count,
                expected_frame_count / float(fps) if fps else 0.0,
            )
        )
        ffmpeg_ok = ffmpeg_meta[0] > 0 and ffmpeg_meta[1] > 0
    except Exception as exc:
        log_info("ffmpeg probe unavailable or failed: {}".format(exc))

    if not reader_ok and not ffmpeg_ok:
        log_info(
            "Video validation failed after writer {}. The file exists, but both imageio "
            "and ffmpeg probes look unhealthy.".format(writer_name)
        )
        return False

    if not ffmpeg_ok:
        log_info(
            "ffmpeg probe did not confirm a positive duration/frame count. "
            "Some players may still reject this MP4 as 0s."
        )
    return True


def resolve_checkpoint_path():
    if cfg.PPO.CHECKPOINT_PATH:
        checkpoint_path = cfg.PPO.CHECKPOINT_PATH
    else:
        checkpoint_path = os.path.join(cfg.OUT_DIR, cfg.ENV_NAME + ".pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Checkpoint file not found: {}. Please specify PPO.CHECKPOINT_PATH "
            "or ensure the trained model exists at the default output path.".format(
                checkpoint_path
            )
        )

    cfg.PPO.CHECKPOINT_PATH = checkpoint_path
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
            if hasattr(first, "state_dict"):
                state_dict = first.state_dict()
            else:
                state_dict = first
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

        for ob_rms_key in ("ob_rms", "obs_rms", "observation_rms"):
            if ob_rms_key in checkpoint:
                ob_rms = checkpoint[ob_rms_key]
                break
    else:
        state_dict = checkpoint

    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    state_dict = maybe_strip_module_prefix(state_dict)
    return state_dict, ob_rms


def load_checkpoint_into_trainer(trainer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(cfg.DEVICE))
    print("Loaded checkpoint object of type: {}".format(type(checkpoint)))

    state_dict, ob_rms = extract_checkpoint_payload(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Unsupported checkpoint payload type: {}. Expected a state_dict, "
            "model object, dict, or [model, ob_rms] tuple/list.".format(
                type(state_dict)
            )
        )

    model_state = trainer.actor_critic.state_dict()
    loadable_state_dict = {}
    skipped_keys = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped_keys.append("{} (missing in current model)".format(key))
            continue
        if model_state[key].shape != value.shape:
            skipped_keys.append(
                "{} (shape mismatch {} != {})".format(
                    key, tuple(value.shape), tuple(model_state[key].shape)
                )
            )
            continue
        loadable_state_dict[key] = value

    if not loadable_state_dict:
        raise RuntimeError(
            "No compatible parameters were found in checkpoint: {}".format(
                checkpoint_path
            )
        )

    missing_keys = sorted(set(model_state.keys()) - set(loadable_state_dict.keys()))
    trainer.actor_critic.load_state_dict(loadable_state_dict, strict=False)

    print(
        "Loaded {} compatible parameter tensors from checkpoint.".format(
            len(loadable_state_dict)
        )
    )
    if skipped_keys:
        preview = ", ".join(skipped_keys[:8])
        if len(skipped_keys) > 8:
            preview += ", ..."
        print("Skipped incompatible checkpoint entries: {}".format(preview))
    if missing_keys:
        preview = ", ".join(missing_keys[:8])
        if len(missing_keys) > 8:
            preview += ", ..."
        print("Model parameters not restored from checkpoint: {}".format(preview))

    if ob_rms is not None:
        set_ob_rms(trainer.envs, ob_rms)
        print("Observation normalization statistics restored from checkpoint.")
    else:
        print(
            "Warning: checkpoint does not contain ob_rms; video observations will "
            "use fresh normalization statistics, which may hurt policy quality."
        )

    trainer.actor_critic.eval()
    return ob_rms


def render_rgb_frame(env):
    try:
        return env.render(mode="rgb_array")
    except TypeError:
        frame = env.render()
        if frame is None:
            raise ValueError(
                "Environment render did not return an RGB frame. "
                "Please verify this env supports render(mode='rgb_array')."
            )
        return frame
    except Exception as exc:
        raise RuntimeError(
            "Environment render failed under mujoco-py. "
            "This often indicates an offscreen OpenGL/context issue: {}".format(exc)
        )


def collect_rollout_frames(env, agent, frame_limit, debug_frame_dir):
    obs = env.reset()
    log_info("Video environment reset complete.")

    frames = []
    first_frame = render_rgb_frame(env)
    debug_frame(first_frame, prefix="captured frame[0]")
    write_debug_image(os.path.join(debug_frame_dir, "frame_000000.png"), first_frame)
    frames.append(first_frame)

    with torch.no_grad():
        for step_idx in range(frame_limit):
            _, act, _ = agent.act(obs)
            obs, _, done, _ = env.step(act)
            frame = render_rgb_frame(env)
            if (
                step_idx < 3
                or step_idx == frame_limit - 1
                or (step_idx + 1) % 100 == 0
            ):
                debug_frame(frame, prefix="captured frame[{}]".format(step_idx + 1))
                log_info(
                    "Collected frame {} / {}".format(
                        step_idx + 2, frame_limit + 1
                    )
                )
                write_debug_image(
                    os.path.join(debug_frame_dir, "frame_{:06d}.png".format(step_idx + 1)),
                    frame,
                )
            frames.append(frame)
            if np.any(done):
                log_info(
                    "Episode ended at step {}; continuing after environment reset."
                    .format(step_idx + 1)
                )

    return frames


def record_with_vec_video_recorder(video_dir, walker_name, trainer, ob_rms, frame_limit):
    video_env = None
    recorder = None
    video_path = get_expected_video_path(video_dir, cfg.ENV_NAME, walker_name)
    debug_frame_dir = os.path.join(
        video_dir, "{}_{}_debug_frames".format(cfg.ENV_NAME, walker_name)
    )
    os.makedirs(debug_frame_dir, exist_ok=True)
    frames = []

    try:
        log_info(
            "Creating video env through make_vec_envs(save_video=True, num_env=1) "
            "to match the repository's built-in recording path."
        )
        video_env = make_vec_envs(
            training=False, norm_rew=False, save_video=True, num_env=1
        )
        if ob_rms is None:
            ob_rms = get_ob_rms(trainer.envs)
        if ob_rms is not None:
            set_ob_rms(video_env, ob_rms)
            log_info("Applied observation normalization statistics to video env.")
        else:
            log_info("Video env is running without observation normalization statistics.")

        recorder = VecVideoRecorder(
            video_env,
            video_dir,
            record_video_trigger=lambda step_id: step_id == 0,
            video_length=frame_limit,
            file_prefix="{}_{}".format(cfg.ENV_NAME, walker_name),
        )
        log_info("VecVideoRecorder initialized with output path {}".format(video_path))

        frames = collect_rollout_frames(recorder, trainer.agent, frame_limit, debug_frame_dir)
        write_debug_gif(
            os.path.join(video_dir, "{}_{}_debug.gif".format(cfg.ENV_NAME, walker_name)),
            frames[: min(len(frames), 200)],
            cfg.VIDEO.FPS,
        )

        recorder.close()
        recorder = None
        cleanup_meta_file(video_path)

        if verify_written_video(
            video_path,
            len(frames),
            cfg.VIDEO.FPS,
            writer_name="VecVideoRecorder",
        ):
            video_path = rewrite_video_for_compatibility(video_path)
            verify_written_video(
                video_path,
                len(frames),
                cfg.VIDEO.FPS,
                writer_name="VecVideoRecorder-compatible",
            )
            return video_path, frames

        log_info(
            "VecVideoRecorder output did not validate cleanly; will fall back to manual frame encoding."
        )
        return video_path, frames
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                pass
        elif video_env is not None:
            try:
                video_env.close()
            except Exception:
                pass


def set_cfg_options():
    calculate_max_iters()
    maybe_infer_walkers()
    calculate_max_limbs_joints()


def calculate_max_limbs_joints():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    num_joints, num_limbs = [], []

    metadata_paths = []
    for agent in cfg.ENV.WALKERS:
        metadata_paths.append(os.path.join(
            cfg.ENV.WALKER_DIR, "metadata", "{}.json".format(agent)
        ))

    for metadata_path in metadata_paths:
        metadata = fu.load_json(metadata_path)
        num_joints.append(metadata["dof"])
        num_limbs.append(metadata["num_limbs"] + 1)

    # Add extra 1 for max_joints; needed for adding edge padding
    cfg.MODEL.MAX_JOINTS = max(num_joints) + 1
    cfg.MODEL.MAX_LIMBS = max(num_limbs) + 1


def calculate_max_iters():
    # Iter here refers to 1 cycle of experience collection and policy update.
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )


def maybe_infer_walkers():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    # Only infer the walkers if this option was not specified
    if len(cfg.ENV.WALKERS):
        return

    xml_dir = os.path.join(cfg.ENV.WALKER_DIR, "xml")
    if not os.path.isdir(xml_dir):
        raise FileNotFoundError(
            "Walker xml directory not found: {}. "
            "Please set --walker-dir or cfg.ENV.WALKER_DIR to the correct location.".format(xml_dir)
        )

    cfg.ENV.WALKERS = [
        xml_file.split(".")[0]
        for xml_file in os.listdir(xml_dir)
        if xml_file.endswith(".xml")
    ]

    if len(cfg.ENV.WALKERS) == 0:
        raise ValueError(
            "No walker xml files found in {}. "
            "Please check the walker directory.".format(xml_dir)
        )


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Generate video for trained RL agent")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "--walker-dir",
        dest="walker_dir",
        help="Override cfg.ENV.WALKER_DIR to find walker xml files",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--walker",
        dest="walker_name",
        help="Specific walker xml name without extension",
        default=None,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See morphology/core/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def generate_video():
    # Ensure walkers are configured before creating environment
    if len(cfg.ENV.WALKERS) == 0:
        print("cfg.ENV.WALKERS is empty, attempting to load walkers from directory...")
        maybe_infer_walkers()

    if len(cfg.ENV.WALKERS) == 0:
        raise ValueError(
            "No walkers found in cfg.ENV.WALKER_DIR: {}. "
            "Please check the directory or use --walker-dir to specify the correct path.".format(
                cfg.ENV.WALKER_DIR
            )
        )

    # For video generation, use only the first walker
    # This is necessary because make_vec_envs only passes xml_file when len(cfg.ENV.WALKERS) == 1
    selected_walker = cfg.ENV.WALKERS[0]
    original_walkers = cfg.ENV.WALKERS.copy()
    cfg.ENV.WALKERS = [selected_walker]

    log_info("Generating video for walker: {}".format(selected_walker))

    su.set_seed(cfg.RNG_SEED)
    log_info(
        "Runtime config: device={}, seed={}, fps={}, video_length={}, out_dir={}".format(
            cfg.DEVICE, cfg.RNG_SEED, cfg.VIDEO.FPS, cfg.PPO.VIDEO_LENGTH, cfg.OUT_DIR
        )
    )
    # Configure the CUDNN backend
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    torch.set_num_threads(1)

    checkpoint_path = resolve_checkpoint_path()
    log_info("Loading checkpoint from: {}".format(checkpoint_path))

    # Create PPO instance without relying on its built-in checkpoint loader so we
    # can support multiple checkpoint formats and restore ob_rms consistently.
    original_checkpoint_path = cfg.PPO.CHECKPOINT_PATH
    cfg.PPO.CHECKPOINT_PATH = ""
    PPOTrainer = PPO(print_model=False)
    cfg.PPO.CHECKPOINT_PATH = original_checkpoint_path
    ob_rms = load_checkpoint_into_trainer(PPOTrainer, checkpoint_path)

    # Create video directory
    video_dir = os.path.join(cfg.OUT_DIR, "videos")
    os.makedirs(video_dir, exist_ok=True)
    original_num_envs = cfg.PPO.NUM_ENVS
    original_vecenv_type = cfg.VECENV.TYPE
    try:
        # ActorCritic.forward() uses cfg.PPO.NUM_ENVS during inference when act=None.
        # Video rollout runs with a single environment, so keep config aligned.
        cfg.PPO.NUM_ENVS = 1
        cfg.VECENV.TYPE = "DummyVecEnv"
        log_info(
            "Temporarily switching rollout config to single-env mode: "
            "NUM_ENVS=1, VECENV.TYPE=DummyVecEnv"
        )
        video_path, frames = record_with_vec_video_recorder(
            video_dir,
            selected_walker,
            PPOTrainer,
            ob_rms,
            cfg.PPO.VIDEO_LENGTH,
        )
        if not verify_written_video(
            video_path,
            len(frames),
            cfg.VIDEO.FPS,
            writer_name="VecVideoRecorder-postcheck",
        ):
            log_info(
                "Falling back to manual frame writer after VecVideoRecorder validation failure."
            )
            save_video_from_frames(frames, video_path, cfg.VIDEO.FPS)
    finally:
        if hasattr(PPOTrainer, "envs") and PPOTrainer.envs is not None:
            try:
                PPOTrainer.envs.close()
            except Exception:
                pass
        cfg.PPO.NUM_ENVS = original_num_envs
        cfg.VECENV.TYPE = original_vecenv_type
        cfg.ENV.WALKERS = original_walkers

    log_info("Video saved to {}".format(video_path))


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    if args.walker_dir is not None:
        cfg.ENV.WALKER_DIR = args.walker_dir

    if args.walker_name is not None:
        cfg.ENV.WALKERS = [args.walker_name]

    # Set cfg options which are inferred
    set_cfg_options()

    if len(cfg.ENV.WALKERS) == 0:
        raise ValueError(
            "cfg.ENV.WALKERS is empty after configuration. "
            "Set --walker or provide valid walker xml files in cfg.ENV.WALKER_DIR."
        )

    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # Save the config
    dump_cfg()
    generate_video()


if __name__ == "__main__":
    main()
