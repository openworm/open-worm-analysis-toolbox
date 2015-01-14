Refactoring Log
===============

Added midbody\_distance as a feature itself, rather than a line of
get\_features\_rewritten in Jim's code, midbody\_distance is the one
piece of data from locomotion that is necessary for the posture
features.

midbody\_distance = locomotion.midbody.speed / config.FPS, giving units
(µm / s) / (frames / s) = µm / frame
