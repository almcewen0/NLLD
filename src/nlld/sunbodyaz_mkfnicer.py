"""
Derive Altitude and Azimuth sun angles with respect to NICER, and
add them to a mkf file
"""

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.units as u

import sys
import argparse

sys.dont_write_bytecode = True


def addAltAzSunToMKF(mkfFile):
    # Open MKF file in update mode
    hduMKF = fits.open(mkfFile, mode='update')
    tbdata = hduMKF['PREFILTER'].data
    hdrPREFILTER = hduMKF['PREFILTER'].header

    # SUN_ANGLE from MKF
    SUN_ANGLE = tbdata.field('SUN_ANGLE')

    # Sun position in RA and DEC
    RA_sun = tbdata.field('SUN_RA')
    DEC_sun = tbdata.field('SUN_DEC')

    # time in NICER MET
    tNICERmkf = tbdata.field('TIME')

    # Quaterninion from MKF
    quaternion_NICER = tbdata.field('QUATERNION')

    ########################
    # Setting up Bore vector
    # vector from NICER frame to star-tracker
    bore = np.array([0.00207192380, 0.00732377336, 0.999971034])

    ##########################################
    # Sun coordinates in NICER reference frame
    vSun_inNICER_all = np.zeros((0, 3))
    sunToBore = np.zeros(len(tNICERmkf))

    for kk in range(0, len(tNICERmkf)):
        sunCoord = SkyCoord(frame='fk5', ra=RA_sun[kk] * u.deg, dec=DEC_sun[kk] * u.deg)
        vSun = sunCoord.cartesian.xyz
        dcm = quatTOdcm(quaternion_NICER[kk])
        # Sun vector in nicer reference frame
        vSun_inNICER = np.matmul(dcm, vSun)
        vSun_inNICER_all = np.vstack((vSun_inNICER_all, vSun_inNICER))
        sunToBore[kk] = np.array(
            np.rad2deg(np.arccos(vSun_inNICER[0] * bore[0] + vSun_inNICER[1] * bore[1] + vSun_inNICER[2] * bore[2])))

    ############################################
    # Correcting sun Theta with Alignment matrix
    # This is from the calibration file
    # nixtipntmis20170601v001.teldef
    rotMatAlign = np.zeros((3, 3))
    rotMatAlign[0, 0] = 9.99997854E-01
    rotMatAlign[0, 1] = -7.58726004E-06
    rotMatAlign[0, 2] = -2.07192380E-03
    rotMatAlign[1, 0] = -7.58726004E-06
    rotMatAlign[1, 1] = 9.99973181E-01
    rotMatAlign[1, 2] = -7.32377336E-03
    rotMatAlign[2, 0] = 2.07192380E-03
    rotMatAlign[2, 1] = 7.32377336E-03
    rotMatAlign[2, 2] = 9.99971034E-01

    vSun_inNICER_all_rotated_alignment = np.zeros((0, 3))
    # Converting Sun vector from NICER coordinates to bore
    # i.e., correcting for bore
    for kk in range(0, len(tNICERmkf)):
        vSun_inNICER_all_rotated_alignment_tmp = np.matmul(rotMatAlign, np.matrix(vSun_inNICER_all[kk, :]).T)
        vSun_inNICER_all_rotated_alignment = np.vstack(
            (vSun_inNICER_all_rotated_alignment, vSun_inNICER_all_rotated_alignment_tmp.T))

    # These should be sun angles, and should be compared to SUN_ANGLE in mkf file for testing purposes    
    sunToZaxis = np.array(np.rad2deg(np.arccos(vSun_inNICER_all[:, 2])))
    sunToZaxis_rotated_alignment = np.array(np.rad2deg(np.arccos(vSun_inNICER_all_rotated_alignment[:, 2])))

    #######################################
    # Plotting All different Sun angles for testing
    fig, ax1 = plt.subplots(1, figsize=(6, 4.5), dpi=80, facecolor='w')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel(r'$\,\mathrm{Time\,(seconds)}$', fontsize=12)
    ax1.set_ylabel('Angles (degrees)', fontsize=12)
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.ticklabel_format(style='plain', axis='both', scilimits=(0, 0), useMathText=True)
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.yaxis.offsetText.set_fontsize(12)

    ax1.plot(tNICERmkf, sunToZaxis, '.', color='black', label='Sun_theta_noCorrection')
    ax1.plot(tNICERmkf, sunToZaxis_rotated_alignment, '.', markersize=18, color='green',
             label='Sun_theta_alignmentCorrected')
    ax1.plot(tNICERmkf, sunToBore, '.', color='blue', label='Sun-toBore')
    ax1.plot(tNICERmkf, SUN_ANGLE, '.', color='r', markersize=1, label='Sun_Angle MKF')

    ax1.legend()

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.tick_params(width=1.5)

    fig.tight_layout()

    plotName = 'sunAnglesVStime.pdf'
    fig.savefig(plotName, format='pdf', dpi=1000)
    plt.close()

    #######################################
    # Calculating the sun Azimuth angle
    sunAzimuth_angle_rotated_alignment = np.array(
        np.rad2deg(np.arctan2(vSun_inNICER_all_rotated_alignment[:, 1], vSun_inNICER_all_rotated_alignment[:, 0])))

    #######################################
    # Plotting Sun Azimuth angles for testing

    fig, ax1 = plt.subplots(1, figsize=(6, 4.5), dpi=80, facecolor='w')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel(r'$\,\mathrm{Time\,(seconds)}$', fontsize=12)
    ax1.set_ylabel('Angles (degrees)', fontsize=12)
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.ticklabel_format(style='plain', axis='both', scilimits=(0, 0), useMathText=True)
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.yaxis.offsetText.set_fontsize(12)

    ax1.plot(tNICERmkf, sunAzimuth_angle_rotated_alignment, '.', color='black', label='Sun_Azimuth')

    ax1.legend()

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.tick_params(width=1.5)

    fig.tight_layout()

    plotName = 'sunAzimuthVStime.pdf'
    fig.savefig(plotName, format='pdf', dpi=1000)
    plt.close()

    ########################
    # adding Sun-Azimuth column to mkf file
    mkfTable = Table.read(mkfFile, format='fits', hdu='PREFILTER')

    AZSUNCol = Column(name='AZ_SUN', data=sunAzimuth_angle_rotated_alignment, unit='deg', format='2.5f')
    mkfTable.add_column(AZSUNCol)

    SUN_X = Column(name='SUN_X', data=vSun_inNICER_all_rotated_alignment[:, 0], unit='', format='2.5f')
    mkfTable.add_column(SUN_X)

    SUN_Y = Column(name='SUN_Y', data=vSun_inNICER_all_rotated_alignment[:, 1], unit='', format='2.5f')
    mkfTable.add_column(SUN_Y)

    SUN_Z = Column(name='SUN_Z', data=vSun_inNICER_all_rotated_alignment[:, 2], unit='', format='2.5f')
    mkfTable.add_column(SUN_Z)

    # Updating event file
    newhdulEF = fits.BinTableHDU(data=mkfTable, header=hdrPREFILTER, name='PREFILTER')
    fits.update(mkfFile, newhdulEF.data, newhdulEF.header, 'PREFILTER')
    ########################
    ########################

    return sunAzimuth_angle_rotated_alignment, SUN_X, SUN_Y, SUN_Z


# Converting quaternion to Direction Cosine Matrix
def quatTOdcm(quat):
    q00 = quat[0] * quat[0]
    q11 = quat[1] * quat[1]
    q22 = quat[2] * quat[2]
    q33 = quat[3] * quat[3]
    q01 = quat[0] * quat[1]
    q23 = quat[2] * quat[3]
    q02 = quat[0] * quat[2]
    q13 = quat[1] * quat[3]
    q12 = quat[1] * quat[2]
    q03 = quat[0] * quat[3]

    dcm = np.zeros((3, 3))
    dcm[0, 0] = q00 - q11 - q22 + q33
    dcm[0, 1] = 2.0 * (q01 + q23)
    dcm[0, 2] = 2.0 * (q02 - q13)
    dcm[1, 0] = 2.0 * (q01 - q23)
    dcm[1, 1] = -q00 + q11 - q22 + q33
    dcm[1, 2] = 2.0 * (q12 + q03)
    dcm[2, 0] = 2.0 * (q02 + q13)
    dcm[2, 1] = 2.0 * (q12 - q03)
    dcm[2, 2] = -q00 - q11 + q22 + q33

    return dcm


def main():
    parser = argparse.ArgumentParser(description="Append Sun Altitude and Azimuth angle to NICER MKF file")
    parser.add_argument("mkfFile", help="A NICER MKF file", type=str)
    args = parser.parse_args()

    addAltAzSunToMKF(args.mkfFile)


if __name__ == '__main__':
    main()
