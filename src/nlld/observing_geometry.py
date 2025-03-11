"""
Module to calculate certain geometric elements related to the ISS and observing sources
Meant  for NICER
"""

import numpy as np
import pandas as pd

import astropy
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import CartesianRepresentation, get_sun, SkyCoord, AltAz, EarthLocation
from astropy.time import Time

from skyfield.positionlib import Geocentric
from skyfield.units import Distance, Velocity
from skyfield.api import load


def iss_islit(vis_windows, issorbitdata):
    """
    Calculates whether iss is in orbit day or night depending on certain times
    :param vis_windows: dataframe with start and end times in isot format, utc (vis_start, vis_end)
    :type vis_windows: pandas.DataFrame
    :param issorbitdata: iss times and coordinate data from OEM ephem file
    :type issorbitdata: pandas.DataFrame
    :return df_islit: string indicating whether visibility window is in orbit day (o_d) or night (o_n) or both (partial)
    :rtype: pandas.DataFrame
    """
    eph = load('de421.bsp')
    islit = np.empty(len(vis_windows), dtype=object)

    vis_windows_duplicates = vis_windows.duplicated(subset=['vis_start', 'vis_end'], keep='first')
    duplicate_indices = vis_windows_duplicates.index[vis_windows_duplicates == True]

    for ii in vis_windows.index:
        # In case this row has already been seen
        if ii in duplicate_indices:
            # Cut dataframe above this row
            vis_windows_tmp = vis_windows.loc[:ii - 1]
            # Get the first occurrence
            vis_start_tmp = vis_windows['vis_start'].loc[ii]
            vis_end_tmp = vis_windows['vis_end'].loc[ii]
            index_first_occurrence = vis_windows_tmp[(vis_windows_tmp['vis_start'] == vis_start_tmp) &
                                                     (vis_windows_tmp['vis_end'] == vis_end_tmp)].index[0]

            islit[ii - vis_windows.index.min()] = islit[index_first_occurrence]
        # These are rows with start and end visibility windows that are new
        else:
            vis_start = vis_windows['vis_start'].loc[ii]
            vis_end = vis_windows['vis_end'].loc[ii]

            issorbitdata_vis_window = issorbitdata[issorbitdata['TIME_UTC'].between(pd.Timestamp(vis_start),
                                                                                    pd.Timestamp(vis_end))]

            islit_vis_window = np.empty(len(issorbitdata_vis_window), dtype=object)
            for jj in issorbitdata_vis_window.index:
                t_iss = Time(issorbitdata_vis_window['TIME_UTC'].loc[jj].strftime('%Y-%m-%dT%H:%M:%S'),
                             format='isot', scale='utc')

                ts = load.timescale()
                time_sf = ts.from_astropy(t_iss)

                ISS_X = Distance(km=issorbitdata_vis_window['ISS_X'].loc[jj]).au
                ISS_Y = Distance(km=issorbitdata_vis_window['ISS_Y'].loc[jj]).au
                ISS_Z = Distance(km=issorbitdata_vis_window['ISS_Z'].loc[jj]).au

                ISS_VX = Velocity(km_per_s=issorbitdata_vis_window['ISS_Vx'].loc[jj]).au_per_d
                ISS_VY = Velocity(km_per_s=issorbitdata_vis_window['ISS_Vy'].loc[jj]).au_per_d
                ISS_VZ = Velocity(km_per_s=issorbitdata_vis_window['ISS_Vz'].loc[jj]).au_per_d

                J2000_mean_ISS = Geocentric([ISS_X, ISS_Y, ISS_Z], [ISS_VX, ISS_VY, ISS_VZ], t=time_sf)

                islit_vis_window[jj - issorbitdata_vis_window.index.min()] = J2000_mean_ISS.is_sunlit(ephemeris=eph)

            if np.all(islit_vis_window):
                islit[ii - vis_windows.index.min()] = 'o_d'
            elif np.all(islit_vis_window == False):
                islit[ii - vis_windows.index.min()] = 'o_n'
            else:
                islit[ii - vis_windows.index.min()] = 'partial'

    vis_windows['orbit'] = islit

    return vis_windows


def bright_earth_angle(iss_cartesian, time, src_ra, src_dec):
    """
    Calculate bright-earth angle based on source and observatory coordinates (for NICER but should work for others)
    Based on Astropy
    :param iss_cartesian: ISS position in Cartesian coordinates (in km)
    :type iss_cartesian: numpy.ndarray
    :param time: astropy TIME object
    :type time: astropy.time.Time
    :param src_ra: Source RA in J2000, degree
    :type src_ra: float
    :param src_dec: Source DEC in J2000, degree
    :type src_dec: float
    :return df_nicer_vis: NICER visibility
    :rtype: pandas.DataFrame
    """
    # Derive the coordinates of the ISS horizon line
    h_ring = iss_horizon_line(iss_cartesian)
    # Place source at completely random distance, which will not matter for our calculation
    src_gcrs_J2000 = SkyCoord(ra=src_ra * u.degree, dec=src_dec * u.degree, distance=2 * u.kpc, frame='icrs')
    # In cartesian coordinates and turn into numpy array
    src_x = src_gcrs_J2000.cartesian.x.to(u.km)
    src_y = src_gcrs_J2000.cartesian.y.to(u.km)
    src_z = src_gcrs_J2000.cartesian.z.to(u.km)
    src_cartesian = np.array([src_x.value, src_y.value, src_z.value])
    # Calculate angle between ISS (I) and source (S) and ISS (I) and edge of horizon line (H)
    vec_IS = np.array([src_cartesian[0] - iss_cartesian[0], src_cartesian[1] - iss_cartesian[1],
                       src_cartesian[2] - iss_cartesian[2]])
    source_to_iss_horizon_angle = np.empty(len(h_ring), dtype=float)

    hring_lit = np.empty(len(h_ring), dtype=bool)
    for nn in range(len(h_ring)):
        # First let's calculate Source to ISS horizon angles
        vec_IH = np.array(
            [h_ring[nn, 0] - iss_cartesian[0], h_ring[nn, 1] - iss_cartesian[1], h_ring[nn, 2] - iss_cartesian[2]])

        # Calculate cosine of the angle
        source_to_iss_horizon_angle[nn] = np.degrees(
            np.arccos(np.dot(vec_IS, vec_IH) / (np.linalg.norm(vec_IS) * np.linalg.norm(vec_IH))))

        # Here we calculate whether Sun is above horizon at h_ring coordinates
        xyz = (h_ring[nn, 0], h_ring[nn, 1], h_ring[nn, 2])  # Xyz coord for each prop. step
        gcrs = coord.GCRS(CartesianRepresentation(*xyz, unit=u.km), obstime=time)  # Let AstroPy know xyz is in GCRS
        itrs = gcrs.transform_to(coord.ITRS(obstime=time))  # Convert GCRS to ITRS
        earth_location = EarthLocation(*itrs.cartesian.xyz)
        # Sun location in Alt-Az
        sun_coord = get_sun(time)
        sun_altaz = sun_coord.transform_to(AltAz(obstime=time, location=earth_location))
        # We are not considering refraction, but if we wish to (pressure=101325*u.Pa, temperature=15*u.deg_C,
        # relative_humidity=0.6)
        # Is location Sun-lit
        hring_lit[nn] = (sun_altaz.alt.deg > 0)

    source_to_iss_horizon = pd.DataFrame({'source_to_iss_horizon_angle': source_to_iss_horizon_angle,
                                          'hring_lit': hring_lit})
    source_to_iss_horizon_od = source_to_iss_horizon[source_to_iss_horizon['hring_lit']]
    bright_earth_angle = source_to_iss_horizon_od['source_to_iss_horizon_angle'].min()

    return bright_earth_angle


def iss_horizon_line(S_iss):
    # See https://physics.stackexchange.com/questions/151388/how-to-calculate-the-horizon-line-of-a-satellite
    R_earth = 6378.137  # (km)
    S_iss_norm = np.linalg.norm(S_iss)
    S_iss_unitvec = S_iss / S_iss_norm

    # Earth to ISS altitude fraction
    R_S_fraction = R_earth / S_iss_norm
    t_ring = np.linspace(0, 2 * np.pi, 100, endpoint=False)

    # Matrix right-side of equation
    mat_x = (-S_iss_unitvec[1] * np.cos(t_ring) - (S_iss_unitvec[0] * S_iss_unitvec[2]) * np.sin(t_ring))
    mat_y = (S_iss_unitvec[0] * np.cos(t_ring) - (S_iss_unitvec[1] * S_iss_unitvec[2]) * np.sin(t_ring))
    mat_z = ((S_iss_unitvec[0] ** 2 + S_iss_unitvec[1] ** 2) * np.sin(t_ring))
    mat_rightside = np.array([mat_x, mat_y, mat_z]).T
    # Matrix multiplier
    mat_multiplier = np.sqrt((1 - R_S_fraction ** 2) / (S_iss_unitvec[0] ** 2 + S_iss_unitvec[1] ** 2))
    # Full right side of equation
    full_rightside = mat_multiplier * mat_rightside

    # Left side of equation
    full_leftside = R_S_fraction * S_iss_unitvec

    # Horizon_line
    h_ring = R_earth * (full_leftside + full_rightside)

    return h_ring


def sunangle(nicertimemjd, srcRA, srcDEC):
    """
    Calculates Sun angle for a source with RA and DEC in degrees J2000
    :param nicertimemjd: numpy array of (nicer) times in MJD
    :type nicertimemjd: numpy.ndarray
    :param srcRA: Right ascension in degrees J2000
    :type srcRA: float
    :param srcDEC: Declination in degrees J2000
    :type srcDEC: float
    :return srcsunangle: array of source sun angles at times nicertimemjd in degrees
    :rtype: numpy.ndarray
    """
    # Define astropy Time instance in mjd format
    nicerTIME = Time(nicertimemjd, format='mjd')

    # Get Sun coordinates
    sunAngGeo = get_sun(nicerTIME)
    sunAngTETE = sunAngGeo.tete

    srcsunangle = np.zeros(len(nicerTIME))
    for jj in range(len(nicerTIME)):
        RA_sun = sunAngTETE[jj].ra.deg
        DEC_sun = sunAngTETE[jj].dec.deg
        srcsunangle[jj] = np.rad2deg(np.arccos(np.sin(np.deg2rad(DEC_sun)) *
                                               np.sin(np.deg2rad(srcDEC)) +
                                               np.cos(np.deg2rad(DEC_sun)) *
                                               np.cos(np.deg2rad(srcDEC)) *
                                               np.cos(np.deg2rad(RA_sun) -
                                                      np.deg2rad(srcRA))))

    return srcsunangle

# Deprecated and less accurate than bright_earth_angle due to the use of circular earth
# def bright_earth_angle_skyfield(iss_cartesian, time, src_ra, src_dec):
#    """
#    Calculate bright-earth angle based on source and observatory coordinates (for NICER but should work for others)
#    Based on Skyfield
#    :param iss_cartesian: ISS position in Cartesian coordinates (in km)
#    :type iss_cartesian: numpy.ndarray
#    :param time: astropy TIME object
#    :type time: astropy.time.Time
#    :param src_ra: Source RA in J2000, degree
#    :type src_ra: float
#    :param src_dec: Source DEC in J2000, degree
#    :type src_dec: float
#    :return df_nicer_vis: NICER visibility
#    :rtype: pandas.DataFrame
#    """
#    # Loading ephemerides
#    eph = load('de421.bsp')

# Define time in skyfield
#    ts = load.timescale()
#    time_sf = ts.from_astropy(time)

# Derive the coordinates of the ISS horizon line
#    h_ring = iss_horizon_line(iss_cartesian)
# Place source at completely random distance, which will not matter for our calculation
#    src_gcrs_J2000 = SkyCoord(ra=src_ra * u.degree, dec=src_dec * u.degree, distance=2 * u.kpc, frame='icrs')
# In cartesian coordinates and turn into numpy array
#    src_x = src_gcrs_J2000.cartesian.x.to(u.km)
#    src_y = src_gcrs_J2000.cartesian.y.to(u.km)
#    src_z = src_gcrs_J2000.cartesian.z.to(u.km)
#    src_cartesian = np.array([src_x.value, src_y.value, src_z.value])
# Calculate angle between ISS (I) and source (S) and ISS (I) and edge of horizon line (H)
#    vec_IS = np.array([src_cartesian[0] - iss_cartesian[0], src_cartesian[1] - iss_cartesian[1],
#                       src_cartesian[2] - iss_cartesian[2]])
#    source_to_iss_horizon_angle = np.empty(len(h_ring), dtype=float)
#    hring_lit = np.empty(len(h_ring), dtype=bool)
#    for nn in range(len(h_ring)):
#        vec_IH = np.array(
#            [h_ring[nn, 0] - iss_cartesian[0], h_ring[nn, 1] - iss_cartesian[1], h_ring[nn, 2] - iss_cartesian[2]])

# Calculate cosine of the angle
#        source_to_iss_horizon_angle[nn] = np.degrees(
#            np.arccos(np.dot(vec_IS, vec_IH) / (np.linalg.norm(vec_IS) * np.linalg.norm(vec_IH))))

#       hring_X_skyfield = Distance(km=h_ring[nn, 0]).au
#       hring_Y_skyfield = Distance(km=h_ring[nn, 1]).au
#       hring_Z_skyfield = Distance(km=h_ring[nn, 2]).au
#       hring_gcrs = Geocentric([hring_X_skyfield, hring_Y_skyfield, hring_Z_skyfield], t=time_sf)
#       hring_lit[nn] = hring_gcrs.is_sunlit(ephemeris=eph)

#   source_to_iss_horizon = pd.DataFrame({'source_to_iss_horizon_angle': source_to_iss_horizon_angle,
#                                         'hring_lit': hring_lit})
#   source_to_iss_horizon_od = source_to_iss_horizon[source_to_iss_horizon['hring_lit']]
#   bright_earth_angle = source_to_iss_horizon_od['source_to_iss_horizon_angle'].min()

#    return bright_earth_angle
