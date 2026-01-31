import logging

from .cli import main

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


if __name__ == "__main__":
    logger.info("Starting RCA CLI entrypoint")
    raise SystemExit(main())
