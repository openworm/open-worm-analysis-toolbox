refactoring log

added midbody_distance as a feature itself, rather than a line of get_features_rewritten
in Jim's code, midbody_distance is the one piece of data from locomotion that is necessary for the posture features.
midbody_distance = locomotion.midbody.speed / config.FPS, giving units (um / s) / (frames / s) = um / frame