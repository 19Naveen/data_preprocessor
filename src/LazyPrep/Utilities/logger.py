import logging
import os

def setup_logger(log_dir='Logs', log_file='pipeline.log', __name__=__name__):
    """
    Set up and configure the logger for the data analysis operations.
    
    Parameters:
    -----------
    log_dir : str, default='Logs'
        Directory where log files will be stored
    log_file : str, default='analyzer.log'
        Name of the log file
        
    Returns:
    --------
    logger : logging.Logger
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')    
    # Configure file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    return logger