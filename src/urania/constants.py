"""Physical constants (SI)."""

import math

G = 6.67430e-11                   # gravitational constant [m³/(kg·s²)], CODATA 2018
DAY = 86400.0                     # [s]
TROPICAL_YEAR = 365.24219 * DAY   # tropical year [s]

# Target nodal rate of a sun-synchronous orbit: 360° per tropical year [rad/s]
SSO_RAAN_RATE = 2.0 * math.pi / TROPICAL_YEAR
