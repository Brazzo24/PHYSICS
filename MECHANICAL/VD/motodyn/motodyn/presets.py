"""Factory functions for plausible baseline bikes."""

from .parameters import (
    MassProperties, Geometry, SuspensionGeometry,
    Powertrain, BrakeSetup, Motorcycle,
)
import math


def generic_sportbike() -> Motorcycle:
    """A mid-size supersport baseline (rider included)."""
    return Motorcycle(
        mass=MassProperties(
            mass=270.0, h_cog=0.60, b=0.70,
            I_roll=35.0, I_pitch=30.0, I_yaw=28.0,
        ),
        geometry=Geometry(
            wheelbase=1.42,
            caster_angle=math.radians(24.0),
            trail=0.095,
        ),
        suspension=SuspensionGeometry(
            swingarm_pivot=(0.40, 0.32),
            sprocket_center=(0.36, 0.27),
            countershaft_sprocket_radius=0.037,
            rear_sprocket_radius=0.090,
            k_rear_wheel=90_000.0,
            k_front_wheel=22_000.0,
        ),
        powertrain=Powertrain(final_drive_ratio=2.7, primary_plus_gearbox=2.1),
        brakes=BrakeSetup(front_brake_fraction=0.75),
        name="generic_sportbike",
    )


def generic_adventure() -> Motorcycle:
    """A tall adventure bike — long travel, high CoG, softer springs."""
    return Motorcycle(
        mass=MassProperties(
            mass=260.0, h_cog=0.75, b=0.72,
            I_roll=38.0, I_pitch=34.0, I_yaw=30.0,
        ),
        geometry=Geometry(
            wheelbase=1.56,
            caster_angle=math.radians(25.5),
            trail=0.105,
        ),
        suspension=SuspensionGeometry(
            swingarm_pivot=(0.44, 0.40),
            sprocket_center=(0.39, 0.33),
            countershaft_sprocket_radius=0.034,
            rear_sprocket_radius=0.098,
            rear_travel=0.210, front_travel=0.210,
            k_rear_wheel=55_000.0,
            k_front_wheel=14_000.0,
        ),
        brakes=BrakeSetup(front_brake_fraction=0.70),
        name="generic_adventure",
    )
