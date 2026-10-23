"""Coverage geometry over a spherical body. SI units. Wertz & Larson, Space Mission Analysis and Design (SMAD),
section 5.2 (spacecraft-centered and Earth-centered angles).

From a satellite at radius R + h: angular radius of the body ρ, sin ρ = R/(R + h); a ground point seen at
elevation ε has nadir angle η with sin η = cos ε sin ρ, and Earth central angle λ = 90° - ε - η.
"""

import math


def angular_radius(altitude: float, radius: float) -> float:
    """Angular radius ρ of the body seen from altitude h: asin(R/(R + h))."""
    return math.asin(radius / (radius + altitude))


def footprint(altitude: float, radius: float, min_elevation: float = 0.0) -> tuple[float, float, float]:
    """Area visible above min_elevation.

    Returns:
        (central angle λ [rad], nadir angle η [rad], slant range D [m]) at the edge
    """
    if altitude <= 0.0:
        raise ValueError(f"Altitude must be positive: {altitude}")
    if not 0.0 <= min_elevation <= math.pi / 2:
        raise ValueError(f"Minimum elevation must be within 0 to 90°: {min_elevation}")
    eta = math.asin(math.cos(min_elevation) * radius / (radius + altitude))
    lam = max(math.pi / 2 - min_elevation - eta, 0.0)
    return lam, eta, _slant_range(altitude, radius, min_elevation)


def _slant_range(altitude: float, radius: float, elevation: float) -> float:
    """Distance from a ground point to a satellite seen at the given elevation (law of cosines)."""
    s = radius * math.sin(elevation)
    return math.sqrt(s * s + altitude * (2.0 * radius + altitude)) - s


def sensor_footprint(altitude: float, radius: float, half_angle: float) -> tuple[float, float, float]:
    """Area seen by a nadir-pointing sensor of half-angle η (clipped at the horizon).

    Returns:
        (central angle λ [rad], elevation ε at the edge [rad], slant range D [m])
    """
    if not 0.0 <= half_angle < math.pi / 2:
        raise ValueError(f"Half-angle must be within 0 to 90°: {half_angle}")
    rho = angular_radius(altitude, radius)
    eta = min(half_angle, rho)
    cos_eps = math.sin(eta) / math.sin(rho)
    eps = math.acos(min(cos_eps, 1.0))
    lam = max(math.pi / 2 - eta - eps, 0.0)
    return lam, eps, _slant_range(altitude, radius, eps)
