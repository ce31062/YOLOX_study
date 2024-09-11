#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

print('Call yolox/exp/__init__.py')

from .base_exp import BaseExp
from .build import get_exp
from .yolox_base import Exp, check_exp_value
