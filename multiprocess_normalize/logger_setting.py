from logging import getLogger, DEBUG, INFO, FileHandler, StreamHandler, Formatter

FILE_FORMAT   = '%(asctime)s - %(name)s - %(message)s'
STREAM_FORMAT = '%(name)s - %(message)s'

def set_logger(logger, log_file=None, level=DEBUG,
               level_fh=DEBUG, level_ch=INFO, 
               fmt_fh=FILE_FORMAT, fmt_ch=STREAM_FORMAT):
    """
    ロガーセットアップのラッパー関数
    """

    logger.setLevel(level)

    if log_file:
        fh = FileHandler(log_file)
        fh.setLevel(level_fh)
        fh_formatter = Formatter(fmt_fh)
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    ch = StreamHandler()
    ch.setLevel(level_ch)
    ch_formatter = Formatter(fmt_ch)
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)