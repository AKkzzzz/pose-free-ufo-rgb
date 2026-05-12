# Copyright (c) Xiaomi Corporation.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .archs import ARCHITECTURES

# Backward-compat alias: top-level ``UFO_models`` still resolves to the
# default ("small") architecture's backbone-size dict.
UFO_models = ARCHITECTURES["small"]
