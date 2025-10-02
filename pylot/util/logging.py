from loguru import logger
import sys

def setup_loguru():
    """
    By default all of the logger outputs will go to stderr for slurm jobs
    This function sets up loguru to also send INFO to stdout so its easier to monitor the job.
    """
    # Remove the default handler (which logs everything to stderr)
    logger.remove()

    # Only INFO → stdout (.out)
    logger.add(
        sys.stdout,
        level="INFO",
        filter=lambda record: record["level"].no == 20  # Only INFO level
    )

    # Everything (including INFO) → stderr (.err)
    logger.add(
        sys.stderr,
        level="DEBUG"
    )