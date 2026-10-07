"""물리 상수 (SI)."""

import math

G = 6.67430e-11                   # 만유인력 상수 [m³/(kg·s²)], CODATA 2018
DAY = 86400.0                     # [s]
TROPICAL_YEAR = 365.24219 * DAY   # 회귀년 [s]

# 태양동기궤도의 목표 승교점 변화율: 1 회귀년에 360° [rad/s]
SSO_RAAN_RATE = 2.0 * math.pi / TROPICAL_YEAR
