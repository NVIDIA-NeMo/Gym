# THIS FILE IS ONLY HERE FOR NEMO-RL INTEGRATION BEFORE WE GO GA.

import sys

import nemo_gym


# Make `penguin` an alias for `nemo_gym`
sys.modules[__name__] = nemo_gym
