import logging
from datetime import datetime
from pathlib import Path

log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"{datetime.now():%m_%d_%Y_%H_%M_%S}.log"

logging.basicConfig(
    filename=log_file,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO
)   

# if __name__ == '__main__':
#     logging.info('Logged successfully!!')