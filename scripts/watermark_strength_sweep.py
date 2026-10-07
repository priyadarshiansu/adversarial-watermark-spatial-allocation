"""Module 4: strength sweep at one mask area (thin wrapper around the area/strength sweep).

    uv run python scripts/watermark_strength_sweep.py --area 0.5 --n-images 10

Same processing, payloads and key as scripts/watermark_area_strength_sweep.py (read from
configs/watermark_calibration.yaml); only the candidate list is replaced by one area and a
list of strengths. Prints the summary; writes nothing.
"""

import argparse

from watermark_area_strength_sweep import (
    DEFAULT_CONFIG,
    load_config,
    print_summary,
    run_sweep,
    summarize,
)

from awsa.utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--area", type=float, default=0.50)
    parser.add_argument("--strengths", type=float, nargs="+",
                        default=[0.01, 0.02, 0.04, 0.06, 0.08, 0.10])
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["candidates"] = [{"area": args.area, "strength": s} for s in args.strengths]

    set_seed(cfg["seed"])
    df = run_sweep(cfg, args.n_images, get_device(args.device))
    print_summary(summarize(df))


if __name__ == "__main__":
    main()
