import logging

logger = logging.getLogger("vad_service")
logger.setLevel(logging.INFO)
log_path = "./vad_service.log"
handler = logging.FileHandler(log_path, mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

