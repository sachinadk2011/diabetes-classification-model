import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:\t  %(message)s"  # Matches Uvicorn's default scannable layout
)
logger = logging.getLogger(__name__)