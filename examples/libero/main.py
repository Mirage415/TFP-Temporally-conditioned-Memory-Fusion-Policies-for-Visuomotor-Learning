import collections
import csv
import dataclasses
import logging
import math
import pathlib
import time
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
CHUNK_TIMING_FIELDNAMES = (
    "task_id",
    "task_description",
    "episode_idx",
    "chunk_idx",
    "planned_steps",
    "executed_steps",
    "infer_sec",
    "exec_sec",
    "episode_step_start",
    "episode_step_end_exclusive",
    "completed_reason",
)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_object"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 20  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

# import random
# replan = random.randint(1, 10)

def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    chunk_timing_csv_path = pathlib.Path(args.video_out_path) / "chunk_timing.csv"
    if chunk_timing_csv_path.exists():
        chunk_timing_csv_path.unlink()

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    if server_metadata.get("temporal_memory_enabled", False):
        effective_replan_steps = int(
            server_metadata.get("tbptt_chunk_len")
            or server_metadata.get("action_horizon")
            or args.replan_steps
        )
    else:
        effective_replan_steps = args.replan_steps
    if effective_replan_steps != args.replan_steps:
        logging.info(
            "Temporal-memory server detected; overriding replan_steps from %d to %d for stepwise hidden-state updates.",
            args.replan_steps,
            effective_replan_steps,
        )

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            client.reset()

            # Setup
            t = 0
            done = False
            chunk_idx = 0
            current_chunk = None
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "timestamp": np.float64(time.monotonic()),
                            "_analysis": {
                                "task_id": int(task_id),
                                "task_description": str(task_description),
                                "task_suite_name": args.task_suite_name,
                                "episode_idx": int(episode_idx),
                                "chunk_idx": int(chunk_idx),
                                "env_step": int(t),
                            },
                        }

                        # Query model to get action
                        infer_start_time = time.perf_counter()
                        action_chunk = client.infer(element)["actions"]
                        infer_latency_sec = time.perf_counter() - infer_start_time
                        planned_steps = min(len(action_chunk), effective_replan_steps)
                        assert (
                            len(action_chunk) >= effective_replan_steps
                        ), f"We want to replan every {effective_replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[:planned_steps])
                        current_chunk = {
                            "chunk_idx": chunk_idx,
                            "planned_steps": planned_steps,
                            "executed_steps": 0,
                            "infer_latency_sec": infer_latency_sec,
                            "episode_step_start": t,
                            "exec_start_time": None,
                        }
                        logging.info(
                            "Chunk ready | task_id=%d episode_idx=%d chunk_idx=%d planned_steps=%d infer_sec=%.4f episode_step_start=%d",
                            task_id,
                            episode_idx,
                            chunk_idx,
                            planned_steps,
                            infer_latency_sec,
                            t,
                        )
                        chunk_idx += 1

                    action = action_plan.popleft()
                    if current_chunk is not None and current_chunk["exec_start_time"] is None:
                        current_chunk["exec_start_time"] = time.perf_counter()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if current_chunk is not None:
                        current_chunk["executed_steps"] += 1
                        if done or not action_plan:
                            completed_reason = "env_done" if done else "chunk_exhausted"
                            _record_chunk_timing(
                                csv_path=chunk_timing_csv_path,
                                task_id=task_id,
                                task_description=task_description,
                                episode_idx=episode_idx,
                                chunk=current_chunk,
                                completed_reason=completed_reason,
                            )
                            current_chunk = None
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    if current_chunk is not None:
                        _record_chunk_timing(
                            csv_path=chunk_timing_csv_path,
                            task_id=task_id,
                            task_description=task_description,
                            episode_idx=episode_idx,
                            chunk=current_chunk,
                            completed_reason="exception",
                        )
                        current_chunk = None
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _record_chunk_timing(
    *,
    csv_path: pathlib.Path,
    task_id: int,
    task_description: str,
    episode_idx: int,
    chunk: dict,
    completed_reason: str,
) -> None:
    exec_start_time = chunk["exec_start_time"]
    exec_sec = 0.0 if exec_start_time is None else time.perf_counter() - exec_start_time
    episode_step_start = int(chunk["episode_step_start"])
    executed_steps = int(chunk["executed_steps"])
    row = {
        "task_id": task_id,
        "task_description": task_description,
        "episode_idx": episode_idx,
        "chunk_idx": int(chunk["chunk_idx"]),
        "planned_steps": int(chunk["planned_steps"]),
        "executed_steps": executed_steps,
        "infer_sec": f"{float(chunk['infer_latency_sec']):.6f}",
        "exec_sec": f"{exec_sec:.6f}",
        "episode_step_start": episode_step_start,
        "episode_step_end_exclusive": episode_step_start + executed_steps,
        "completed_reason": completed_reason,
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CHUNK_TIMING_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logging.info(
        "Chunk timing | task_id=%d episode_idx=%d chunk_idx=%d planned_steps=%d executed_steps=%d infer_sec=%s exec_sec=%s env_step_range=[%d,%d) reason=%s",
        row["task_id"],
        row["episode_idx"],
        row["chunk_idx"],
        row["planned_steps"],
        row["executed_steps"],
        row["infer_sec"],
        row["exec_sec"],
        row["episode_step_start"],
        row["episode_step_end_exclusive"],
        row["completed_reason"],
    )


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
