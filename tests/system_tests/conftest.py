# test/conftest.py
import os
import sys

import pytest

# 添加源代码路径到 PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
