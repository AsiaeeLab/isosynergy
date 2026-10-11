"""Rounded empirical curve parameters generated from the public replicate measurements."""
import paths
import json
import numpy as np
_data=json.loads((paths.INPUTS/'noise_model_curves.json').read_text())
VIABILITY_MEAN=np.array(_data['viability_mean'])
VIABILITY_SD=np.array(_data['viability_sd'])
VIABILITY_ROBUST_SD=np.array(_data['viability_robust_sd'])
POSITIVE_MEAN=np.array(_data['positive_noise_mean'])
POSITIVE_LOG_SD=np.array(_data['positive_noise_log_sd'])
