"""
Configuration Package
====================

This package handles all configuration settings for the application.
"""

import logging

# 配置包的日志记录器
logger = logging.getLogger(__name__)

def _setup_logging(path: str = None) -> logging.Logger:
    """设置日志记录器
    
    Args:
        path: 日志配置文件路径
        
    Returns:
        logging.Logger: 日志记录器
    """
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler('math_solver.log', mode='w')
    file_handler.setFormatter(formatter)
    
    # 设置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 移除所有现有的处理器
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # 添加新的处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 初始化日志配置
_setup_logging()

__all__ = ['_setup_logging']
