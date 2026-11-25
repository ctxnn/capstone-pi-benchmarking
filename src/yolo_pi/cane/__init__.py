"""Hardware drivers and the runtime that turns the toolkit into a working cane.

The rest of ``yolo_pi`` is hardware-free: types, fusion policy, detector and
camera source. This package adds what runs on the cane itself:

* ``bus``, ``tof``, ``ultrasonic``, ``imu``, ``haptics``: device drivers;
* ``sensors``: threads that turn raw reads into fusion ``SensorReading`` values;
* ``priority``: close range (ToF + ultrasonic) vs far range (YOLO) vs overhead;
* ``vision``: camera + detector thread feeding detections to fusion;
* ``runtime`` and ``app``: the fuse-decide-vibrate loop and its assembly;
* ``sim``: a simulated cane for laptop runs and tests;
* ``cli``: the ``yolo-pi-cane`` command.

Nothing here imports Raspberry Pi libraries at module load; they are imported
when a real device is built, so the package imports (and its tests run) anywhere.
"""

from .config import CaneConfig, ConfigError, load_config

__all__ = ["CaneConfig", "ConfigError", "load_config"]
