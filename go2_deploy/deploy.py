import argparse
import pickle
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Deploy a policy to a Unitree Go2.")
    parser.add_argument(
        "interface", nargs="?", help="Robot network interface, e.g. eth0"
    )
    parser.add_argument("--policy-checkpoint", help="Exact model_*.pt file to use")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check a policy offline without the SDK",
    )
    args = parser.parse_args()
    if args.preflight and not args.policy_checkpoint:
        parser.error("--preflight requires --policy-checkpoint")
    if not args.preflight and not args.interface:
        parser.error("a robot network interface is required for deployment")

    policy = None
    if args.policy_checkpoint:
        from go2_deploy.rl_controller import RLController

        try:
            policy = RLController(args.policy_checkpoint)
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            pickle.UnpicklingError,
        ) as exc:
            parser.error(f"Policy preflight failed: {exc}")
        print(f"Policy checkpoint: {policy.checkpoint_path}")
        print(f"Policy SHA-256: {policy.checkpoint_sha256}")
        cfg = policy.cfg
        print(f"Controller defaults: {cfg.ctrl_freq} Hz, Kp {cfg.kp}, Kd {cfg.kd}")
        print(f"Action EMA alpha: {cfg.ema_smoothing_factor}")
    if args.preflight:
        print("Offline checkpoint load and inference passed. No robot connection used.")
        return

    if sys.platform != "linux":
        parser.error("Go2 deployment requires Linux (timerfd).")

    try:
        from unitree_sdk2py.core import channel
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except ModuleNotFoundError as exc:
        if exc.name != "unitree_sdk2py":
            raise
        parser.error(
            "Unitree SDK is not installed. Follow README_DEPLOY.md, then run "
            "uv run --frozen --extra unitree_sdk python -m go2_deploy.deploy "
            "<interface>."
        )

    from go2_deploy.main_controller import MainController
    from go2_deploy.keyboard_handler import KeyboardHandler

    # CycloneDDS 0.10.2 bug workaround
    channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
        "<Verbosity>config</Verbosity>", "<Verbosity>none</Verbosity>"
    )

    ChannelFactoryInitialize(
        0,
        args.interface,
    )
    controller = MainController(rl_controller=policy)
    keyboard_handler = KeyboardHandler(controller)  # noqa: F841

    print(
        "\nConnection successful! Keep the robot away from obstacles and "
        + "always have access to the remote controller\n"
    )

    try:
        while True:
            time.sleep(1)
    finally:
        controller.csv_logger.close()


if __name__ == "__main__":
    main()
