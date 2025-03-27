import numpy as np
import pandas as pd
import argparse
import os
import datetime as dt
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from glob import glob
from subprocess import call

def build_catalogs(survey='both'):
    """
    Read source data from full survey files. If files don't
    exist, attempt to download. A pared down catalog is produced from these, 
    which will be read in the future. These catalogs are hard-
    coded to omit sources without published fluxes above 1e-12 erg/s/cm^2 
    in any observing band, and sources within 3 arcminutes of known NICER
    targets are also omitted. Missing values are filtered out of all 
    catalogs.
    """


    nicer_targets = pd.read_csv("source_catalogs/NICER2_Xray_Target_List_V42t.csv",skiprows=2)
    nicer_mask = ~np.isnan(nicer_targets['RAJ_DEG'])
    s_nicer = SkyCoord(nicer_targets['RAJ_DEG'][nicer_mask],
                       nicer_targets['DECJ_DEG'][nicer_mask],
                       unit='deg',frame='icrs')

    if survey in ['both','4xmm']:
        if not os.path.isfile("source_catalogs/4XMM_DR13cat_v1.0_full.fits"):
            print("Downloading 4XMM catalog")
            call("curl -o 4XMM_DR13cat_v1.0.fits.gz" + \
            "http://xmmssc.irap.omp.eu/Catalogue/4XMM-DR13/4XMM_DR13cat_v1.0.fits.gz",shell=True)
            call("tar xf 4XMM_DR13cat_v1.0.fits.gz; rm 4XMM_DR13cat_v1.0.fits.gz",shell=True)
            call("mv 4XMM_DR13cat_v1.0* source_catalogs/4XMM_DR13cat_v1.0_full.fits",shell=True)
            
        catalog_4xmm_dr13 = Table.read("source_catalogs/4XMM_DR13cat_v1.0_full.fits")
        s_4xmm = SkyCoord(catalog_4xmm_dr13['RA'],
                          catalog_4xmm_dr13['DEC'],
                          unit='deg',frame='icrs')

        _, ang_sep, _ = s_4xmm.match_to_catalog_sky(s_nicer)
        catalog_4xmm_dr13 = catalog_4xmm_dr13[ang_sep > 3*u.arcmin]

        bands = np.array(catalog_4xmm_dr13.colnames)[
                         np.array(
                             ['FLUX' in n and 'PN_' in n and 'ERR' not in n for n in catalog_4xmm_dr13.colnames]
                             )
                         ]
        fluxes = np.max([catalog_4xmm_dr13[b] for b in bands],axis=0)
        catalog_4xmm_dr13 = catalog_4xmm_dr13[fluxes>1e-12]
        catalog_4xmm_dr13.add_column(fluxes[fluxes>1e-12],name='MAXFLUX')

        tmp = open("source_catalogs/4XMM_targets.csv",'w')
        tmp.write("id,src,ra,dec,name,propnum,targnum,maxflux\n")
        for ln in catalog_4xmm_dr13:
            tmp.write(f"{ln['SRCID']},{ln['IAUNAME'].replace(' ','_')}," + \
                      f"{ln['RA']},{ln['DEC']},OID{ln['OBS_ID']},,,{ln['MAXFLUX']}\n")
        tmp.close()


    if survey in ['both','erosita']:
        if not os.path.isfile("source_catalogs/eRASS1_Main.v1.1.fits"):
            print("Downloading main eROSITA catalog")
            call("curl -o eRASS1_Main.tar.gz " + \
            "https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/MerloniA_DR1/eRASS1_Main.tar.gz"
                 ,shell=True)
            call("tar xf eRASS1_Main.tar.gz; rm eRASS1_Main.tar.gz",shell=True)
            call("mv eRASS1_Main.v1.1.fits source_catalogs",shell=True)
        er_main = Table.read("source_catalogs/eRASS1_Main.v1.1.fits")
        s_erm = SkyCoord(er_main['RA'],er_main['DEC'],unit='deg',frame='icrs')
        _, ang_sep, _ = s_erm.match_to_catalog_sky(s_nicer)
        er_main = er_main[ang_sep > 3*u.arcmin]

        bands = np.array(er_main.colnames)[np.array(['FLUX' in n and 'ERR' not in n for n in er_main.colnames])]
        fluxes_erm = np.max([er_main[b] for b in bands],axis=0)
        er_main = er_main[fluxes_erm>1e-12]
        fluxes_erm = fluxes_erm[fluxes_erm>1e-12]
        er_main.add_column(fluxes_erm,name='MAXFLUX')


        if not os.path.isfile("source_catalogs/eRASS1_Hard.v1.0.fits"):
            print("Downloading eROSITA hard file")
            call("curl -o eRASS1_Hard.tar.gz " + \
            "https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/MerloniA_DR1/eRASS1_Hard.tar.gz"
                 ,shell=True)
            call("tar xf eRASS1_Hard.tar.gz; rm eRASS1_Hard.tar.gz",shell=True)
            call("mv eRASS1_Hard.v1.0.fits source_catalogs ",shell=True)
        er_hard = Table.read("source_catalogs/eRASS1_Hard.v1.0.fits")
        s_erh = SkyCoord(er_hard['RA'],er_hard['DEC'],unit='deg',frame='icrs')
        _, ang_sep, _ = s_erh.match_to_catalog_sky(s_nicer)
        er_hard = er_hard[ang_sep > 3*u.arcmin]

        bands = np.array(er_hard.colnames)[np.array(['FLUX' in n and 'ERR' not in n for n in er_hard.colnames])]
        fluxes_erh = np.max([er_hard[b] for b in bands],axis=0)
        er_hard = er_hard[fluxes_erh>1e-12]
        fluxes_erh = fluxes_erh[fluxes_erh>1e-12]
        er_hard.add_column(fluxes_erh,name='MAXFLUX')  

        tmp = open("source_catalogs/eROSITA_targets.csv",'w')
        tmp.write("id,src,ra,dec,name,propnum,targnum,maxflux\n")
        for tab in [er_main,er_hard]:
            for ln in tab:
                tmp.write(f"{ln['SKYTILE']}+{ln['ID_SRC']},{ln['IAUNAME'].replace(' ','_')}," + \
                          f"{ln['RA']},{ln['DEC']},,,,{ln['MAXFLUX']}\n")
        tmp.close()


def filter_data(ra,dec,region,region_size,fluxlim=1e-12,build=False):
    """
    Filter survey catalogs to find sources in a provided region.

    Inputs
    ======
    ra, dec [float] : coordinates [degrees] of the region's center point
    region [str] : either 'circle' or 'ra_strip', where the latter covers all 
                   declinations within ra-region_size <= ra <= ra+region_size
    region_size [float] : either the radius or the width of the strip in
                          arcminutes
    fluxlim [float] : minimum catalog source flux [erg/s/cm^2] to be included 
                       in target list
    build [bool] : if True and source catalogs don't exist, will attempt to
                   download and build them

    Returns
    =======
    outdata [dict] : contains matching sources for each of the included survey files
    """
    center = SkyCoord(ra,dec,frame='icrs',unit='deg')
    outdata = {}
    for fil in ["source_catalogs/eROSITA_targets.csv","source_catalogs/4XMM_targets.csv"]:
        if not os.path.isfile(fil):
            if build:
                build_catalogs(fil.split('/')[-1].split('_')[0].lower())
            else:
                print(f"{fil} doesn't exist and build is off; skipping")
                continue

        subtab = pd.read_csv(fil)
        s = SkyCoord(subtab['ra'],subtab['dec'],frame='icrs',unit='deg')
        if region == 'circle':
            pos_cnd = s.separation(center) < region_size*u.arcmin
        elif region == 'ra_strip':
            pos_cnd = (s.ra-center.ra)*np.cos(s.dec.to('rad')) < region_size*u.arcmin
        else:
            raise ValueError("Region must be either 'circle' or 'ra_strip'")
        flux_cnd = subtab['maxflux'] >= fluxlim
        cnd = flux_cnd & pos_cnd
        outdata[fil.rstrip('csv')] = subtab[cnd]

    return outdata


    
def make_target_files(data,outfile):
    """
    Write out target lists in NICER format.

    Inputs
    ======
    data [dict] : contains sources to be written (this is the output from
                  filter_data)
    outfile [str] : prefix of the output file list; survey and today's date
                    will also be added to this name
    """
    targetfile_preface = "NICER/SEXTANT/OHMAN Target IDs and Names,,,,,,\n" + \
                         "Flight v42s;3/15/25,,,,,,\n" + \
                         "ID,Source,RAJ_DEG,DECJ_DEG,Other Names,Proposal number,Target number\n"
    today = dt.datetime.today()
    today = f"{str(today.month).zfill(2)}{str(today.day).zfill(2)}{today.year}"
    for key in data.keys():
        if len(data[key]) == 0:
            print(f"No sources in {key.rstrip('.')} are visible")
            continue
        else:
                tmp = open(f"{outfile}_{key.split('/')[-1].split('_')[0]}_{today}.csv",'w')
                tmp.write(targetfile_preface)
                for ln in data[key].to_numpy():
                    tmp.write(f"{ln[0]},{ln[1]},{ln[2]},{ln[3]},,\n")
        tmp.close()




def main():
    parser = argparse.ArgumentParser(
                 description="Generate target lists for NICER using X-ray source catalogs")

    parser.add_argument("-ra", 
                        help="Center RA for target list [deg]", 
                        required=True,
                        type=float
                        )
    parser.add_argument("-dec", 
                        help="Center Declination for target list [deg]", 
                        required=True,
                        type=float
                        )
    parser.add_argument("-reg",
                        help="Shape of target region ['circle','ra_strip'], default circle",
                        default='circle',
                        type=str
                        )

    parser.add_argument("-s",
                       help='Region size in arcminutes, default 15. If region ' + \
                            'is circular, this corresponds to the radius; ' + \
                            'otherwise, it is the width of the strip',
                       type=float,
                       default=15
                       )

    parser.add_argument("-f",
                        help="Output file name prefix (will be appended with each" + \
                             "catalog name and csv extension)",
                        default='targets',
                        type=str
                        )

    parser.add_argument("-fl",
                        help="Flux limit in ergs/s/cm^2 (Note: catalogs have" + \
                             "already been stripped of sources below 1e-12)",
                        default=1e-12,
                        type=float
                        )
    parser.add_argument("-b",
                        help="Flag to build databases from scratch if they" + \
                        "don't already exist",
                        type=bool,
                        default=False,
                        action = argparse.BooleanOptionalAction
                        )


    args = parser.parse_args()
    data = filter_data(args.ra,args.dec,args.reg,args.s,args.fl,args.b)
    make_target_files(data,args.f) 

if __name__ == "__main__":
    main()
