import logging

from dotenv import dotenv_values

env_config = dotenv_values()


# Configure logger
logger = logging.getLogger('mdc-logger')
logger.setLevel(env_config['LOG_LEVEL'])

# Get handler
handler = logging.StreamHandler()
handler.setLevel(env_config['LOG_LEVEL'])

# Get formatter
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)
