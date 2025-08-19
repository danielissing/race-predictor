"""
Geographic and geometric utility functions.
Handles distance calculations and map-related operations.
"""

import math
import numpy as np
import config

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Uses the Haversine formula to calculate the shortest distance between
    two points on a sphere (Earth), measured along the surface.

    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)

    Returns:
        Distance in meters

    Example:
        # Distance from NYC to London
        haversine_m(40.7128, -74.0060, 51.5074, -0.1278) -> ~5,570,000m
    """
    # Convert decimal degrees to radians
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return config.EARTH_R * c


def aid_station_markers(gpx_df, aid_km, cluster_radius_m: float = config.CLUSTER_RADIUS):
    """
    Create clustered map markers for aid stations from GPX data.

    Aid stations that are close together (within cluster_radius_m) are grouped
    into a single marker to avoid map clutter. This is useful when an aid station
    appears multiple times (e.g., in an out-and-back or loop course).

    Args:
        gpx_df: DataFrame with 'lat', 'lon', 'dist_m' columns
        aid_km: List of cumulative distances to aid stations (in km)
        cluster_radius_m: Maximum distance to group stations together

    Returns:
        List of marker dictionaries with:
            - lat, lon: Marker position
            - labels: List of aid station names (e.g., ['AS1', 'AS4'])
            - kms: List of distances for each station

    Example:
        If AS1 at 10km and AS4 at 42km are at the same physical location:
        [{'lat': 45.5, 'lon': -122.6, 'labels': ['AS1','AS4'], 'kms': [10.0, 42.2]}]
    """
    markers = []

    # Create individual markers for each aid station
    for i, km in enumerate(aid_km, start=1):
        target_distance_m = km * 1000.0

        # Find the closest GPS point to this distance
        idx = int(np.searchsorted(gpx_df['dist_m'].values, target_distance_m, side='right'))
        idx = max(0, min(idx, len(gpx_df) - 1))

        lat = float(gpx_df['lat'].iloc[idx])
        lon = float(gpx_df['lon'].iloc[idx])

        markers.append({
            'lat': lat,
            'lon': lon,
            'label': f"AS{i}",
            'km': float(km)
        })

    # Cluster nearby markers
    clusters = []
    for marker in markers:
        placed = False

        # Check if this marker is close to any existing cluster
        for cluster in clusters:
            distance = haversine_m(
                marker['lat'], marker['lon'],
                cluster['lat'], cluster['lon']
            )

            if distance <= cluster_radius_m:
                # Add to existing cluster
                cluster['labels'].append(marker['label'])
                cluster['kms'].append(marker['km'])
                placed = True
                break

        if not placed:
            # Create new cluster
            clusters.append({
                'lat': marker['lat'],
                'lon': marker['lon'],
                'labels': [marker['label']],
                'kms': [marker['km']]
            })

    return clusters