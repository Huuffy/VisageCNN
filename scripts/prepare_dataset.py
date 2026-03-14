"""Master dataset preparation orchestrator for VisageCNN.

Runs the full data pipeline in one command:
  1. Download  -- HuggingFace emotion datasets (fer2013, rafdb, affectnet, expw)
  2. Import    -- CK+48/ local folder if present
  3. Filter    -- confidence-based quality filter (requires: pip install transformers)
  4. Balance   -- cap majority classes to MAX_RATIO x minority class

Usage
-----
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --sources rafdb affectnet
    python scripts/prepare_dataset.py --skip-download   # import + filter + balance only
    python scripts/prepare_dataset.py --skip-filter     # skip quality filter
    python scripts/prepare_dataset.py --max-ratio 4.0
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from visage_er.config import Config

SCRIPTS_DIR = Path(__file__).parent
ALL_HF_SOURCES = ['fer2013', 'rafdb', 'affectnet', 'expw']
CK_PLUS_PATH = Path(__file__).parent.parent / 'CK+48'


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def run_step(label: str, cmd: list, logger: logging.Logger) -> bool:
    """Run a sub-script as a subprocess and stream its output.

    Args:
        label: Human-readable step name for logging.
        cmd: Command list passed to subprocess.run.
        logger: Logger instance.

    Returns:
        True if the step exited with code 0, False otherwise.
    """
    print(f"\n{'=' * 60}")
    print(f"STEP: {label}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.warning(f"Step '{label}' exited with code {result.returncode} -- continuing.")
        return False
    return True


def check_transformers() -> bool:
    """Return True if the transformers package is importable."""
    try:
        import transformers
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="VisageCNN full dataset preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Steps run in order:\n"
            "  1. download    HuggingFace emotion datasets\n"
            "  2. import      CK+48/ local folder (auto-detected)\n"
            "  3. filter      confidence-based quality filter  (needs: pip install transformers)\n"
            "  4. balance     cap majority classes to --max-ratio x minority\n"
        ),
    )
    parser.add_argument(
        '--sources', nargs='+', default=ALL_HF_SOURCES,
        choices=ALL_HF_SOURCES, metavar='SOURCE',
        help=f"HuggingFace sources to download (default: all). Options: {ALL_HF_SOURCES}",
    )
    parser.add_argument(
        '--skip-download', action='store_true',
        help="Skip the download step (useful when data is already present)",
    )
    parser.add_argument(
        '--skip-filter', action='store_true',
        help="Skip the quality filter step",
    )
    parser.add_argument(
        '--filter-threshold', type=float, default=0.70,
        help="Confidence threshold for the quality filter (default: 0.70)",
    )
    parser.add_argument(
        '--max-ratio', type=float, default=4.0,
        help="Max class imbalance ratio for the balance step (default: 4.0)",
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.2,
        help="Val split fraction for download + import steps (default: 0.2)",
    )
    parser.add_argument(
        '--seed', type=int, default=42,
    )
    args = parser.parse_args()

    logger = setup_logging()
    Config.create_directories()
    py = sys.executable

    print("\n" + "=" * 60)
    print("VISAGECNN DATASET PREPARATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Download
    # ------------------------------------------------------------------
    if not args.skip_download:
        run_step(
            "Download HuggingFace sources",
            [
                py, str(SCRIPTS_DIR / 'download_data.py'),
                '--sources', *args.sources,
                '--val-ratio', str(args.val_ratio),
                '--seed', str(args.seed),
            ],
            logger,
        )
    else:
        logger.info("Step 1 (download) skipped.")

    # ------------------------------------------------------------------
    # Step 2 — Import CK+
    # ------------------------------------------------------------------
    if CK_PLUS_PATH.exists():
        run_step(
            "Import CK+48",
            [
                py, str(SCRIPTS_DIR / 'import_ckplus.py'),
                '--src', str(CK_PLUS_PATH),
                '--val-ratio', str(args.val_ratio),
                '--seed', str(args.seed),
            ],
            logger,
        )
    else:
        logger.info(
            f"Step 2 (CK+ import) skipped -- CK+48/ not found at {CK_PLUS_PATH}.\n"
            f"  Place the extracted CK+48 folder at {CK_PLUS_PATH} to include it."
        )

    # ------------------------------------------------------------------
    # Step 3 — Quality filter
    # ------------------------------------------------------------------
    if args.skip_filter:
        logger.info("Step 3 (quality filter) skipped by --skip-filter flag.")
    elif not check_transformers():
        logger.warning(
            "Step 3 (quality filter) skipped -- 'transformers' not installed.\n"
            "  Install it with:  pip install transformers\n"
            "  Then run manually: python scripts/filter_dataset.py"
        )
    else:
        run_step(
            "Quality filter (confidence-based label validation)",
            [
                py, str(SCRIPTS_DIR / 'filter_dataset.py'),
                '--threshold', str(args.filter_threshold),
            ],
            logger,
        )

    # ------------------------------------------------------------------
    # Step 4 — Balance
    # ------------------------------------------------------------------
    run_step(
        "Balance classes (cap majority)",
        [
            py, str(SCRIPTS_DIR / 'balance_dataset.py'),
            '--cap',
            '--max-ratio', str(args.max_ratio),
            '--seed', str(args.seed),
        ],
        logger,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nDataset: {Config.DATASET_PATH}")
    print("\nNext steps:")
    print("  1. Delete old cache:   rmdir /s /q models\\cache")
    print("  2. Start training:     python scripts/train_hybrid.py")
    print("\nTo re-run without re-downloading:")
    print("  python scripts/prepare_dataset.py --skip-download")
    print("\nTo restore capped or filtered images:")
    print("  python scripts/balance_dataset.py --restore")
    print("  python scripts/filter_dataset.py --restore")
    print("=" * 60)


if __name__ == "__main__":
    main()
