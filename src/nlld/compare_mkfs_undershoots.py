"""
Compare the undershoots in N sets of MKF files
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nlld.nicermkf import MkfFileOps, define_nicerdetloc
from nlld.mkfdiagnostics import mkf_diagnostics, creatediagnosticsplots_od, creatediagnosticsplots_on
from nlld.nlld_logging import get_logger

import sys
import argparse
import os

sys.dont_write_bytecode = True

# Log config
############
logger = get_logger(__name__)


def comparemkfundershoots(listofmkffiles, sunshine=2, sunAngLR=45, sunAngUR=180, moonAngLR=0, moonAngUR=180,
                          moonphaseLR=0, moonphaseUR=1, brearthLR=0, brearthUR=180, sunAzLR=-180, sunAzUR=180,
                          timepostleak=True, under='MPU_UNDERONLY_COUNT', largestfpms_to_flag=0, diagnosticsplots=False,
                          outputdir='compare_mkfs'):
    logger.info('\n largestfpms_to_flag : ' + str(largestfpms_to_flag))
    logger.info('\n Output directory : ' + str(outputdir))

    # Create folder in working directory to drop plots and files in
    if not os.path.exists(outputdir):
        # if the parent directory is not present, create it
        os.makedirs(outputdir, exist_ok=True)

    # Create working directory for specific cuts
    try:
        directory_specific_cuts = ('ss' + str(sunshine) + '_sl' + str(sunAngLR) + '_su' + str(sunAngUR) + '_ml'
                                   + str(moonAngLR) + '_mu' + str(moonAngUR) + '_mpl' + str(moonphaseLR) + '_mpu' +
                                   str(moonphaseUR) + '_bel' + str(brearthLR) + '_beu' + str(brearthUR) + '_al' +
                                   str(sunAzLR) + '_au' + str(sunAzUR) + '_lff' + str(largestfpms_to_flag) + '_'
                                   + under)
        os.makedirs(directory_specific_cuts)
        command = 'mv ' + directory_specific_cuts + ' ' + outputdir
        os.system(command)
    except FileExistsError:
        print(f"Directory '{directory_specific_cuts}' already exists.")

    # Reading detectors for later comparison purposes
    nicDET_geograph = define_nicerdetloc()

    # initializing merged dataframe of all MKF files in list
    merged_average_undershoot_perFPM = pd.DataFrame()
    for kk, mkffile in enumerate(listofmkffiles):

        filteredmkf, average_undershoot_perFPM, _ = mkf_diagnostics(mkffile, sunshine=sunshine, sunAngLR=sunAngLR,
                                                                    sunAngUR=sunAngUR, moonAngLR=moonAngLR,
                                                                    moonphaseLR=moonphaseLR, moonphaseUR=moonphaseUR,
                                                                    moonAngUR=moonAngUR, brearthLR=brearthLR,
                                                                    brearthUR=brearthUR, sunAzLR=sunAzLR,
                                                                    sunAzUR=sunAzUR, timepostleak=timepostleak,
                                                                    under=under)

        # Move on if mkf is empty after filtering
        if filteredmkf.empty:
            logger.info('{} file: DataFrame after all filtering is empty - moving on'.format(mkffile))
            continue

        # Suffix for several file names
        suffix = mkffile.split(".")[0]

        # Create all diagnostics plots and move them to their own directory 'outputdir'
        if diagnosticsplots:
            try:
                os.makedirs(suffix + '_diagnostics_plots')
                command = 'mv ' + suffix + '_diagnostics_plots ' + outputdir + '/' + directory_specific_cuts
                os.system(command)
            except FileExistsError:
                print(f"Directory diagnostics_plots already exists.")
            # Depending on filtering
            if sunshine == 1:
                creatediagnosticsplots_od(filteredmkf, average_undershoot_perFPM, nicDET_geograph, suffix)
            elif sunshine == 0:
                creatediagnosticsplots_on(filteredmkf, average_undershoot_perFPM, nicDET_geograph, suffix)
            else:
                raise Exception('Can only produce diagnostics plots for orbit day or orbit night separately')
            command = 'mv ' + suffix + '*.png ./' + outputdir + '/' + directory_specific_cuts + '/' + suffix + '_diagnostics_plots/'
            os.system(command)

        # Merging the mkf files
        if merged_average_undershoot_perFPM.empty:
            # First get nan indices of averaged array
            merged_nan_indices = average_undershoot_perFPM.loc[(average_undershoot_perFPM['median'].isna())].index
            # Change name of columns in preparation of merging them with the rest of the files
            merged_average_undershoot_perFPM = average_undershoot_perFPM
            merged_average_undershoot_perFPM = merged_average_undershoot_perFPM.rename(
                columns={'average': 'average_' + str(kk), 'stdv': 'stdv_' + str(kk), 'median': 'median_' + str(kk)})
        else:
            # Get nan indices of subsequent files and merge with nan indices from first one
            nan_indices = average_undershoot_perFPM.loc[(average_undershoot_perFPM['median'].isna())].index
            merged_nan_indices = merged_nan_indices.append(nan_indices).drop_duplicates(keep='first')
            # Merge all average_values dataframe with one another
            merged_average_undershoot_perFPM = pd.merge(merged_average_undershoot_perFPM, average_undershoot_perFPM,
                                                        how='outer', left_index=True, right_index=True)
            merged_average_undershoot_perFPM = merged_average_undershoot_perFPM.rename(
                columns={'average': 'average_' + str(kk), 'stdv': 'stdv_' + str(kk), 'median': 'median_' + str(kk)})

    # If all months resulted in no data, i.e., we never sampled that parameters space
    if merged_average_undershoot_perFPM.empty:
        logger.info('This specific parameter space cut was never sampled - returning empty dataframes')
        merged_median_undershoot_perFPM_allclean = merged_average_undershoot_perFPM_allclean = \
            corr_matrix_median = corr_matrix_average = indices_good_det = pd.DataFrame()
        command = 'mv *.log ./' + outputdir + '/' + directory_specific_cuts
        os.system(command)
        return (merged_median_undershoot_perFPM_allclean, merged_average_undershoot_perFPM_allclean,
                corr_matrix_median, corr_matrix_average, indices_good_det)

    # Drop all nans
    logger.info('Detectors removed due to NaN {}'.format(merged_nan_indices.to_list()))
    merged_average_undershoot_perFPM_nonan = merged_average_undershoot_perFPM.drop(merged_nan_indices)

    if merged_average_undershoot_perFPM_nonan.empty:
        logger.info('This specific parameter space cut resulted in the excision of all detectors - '
                    'returning empty dataframes')
        merged_median_undershoot_perFPM_allclean = merged_average_undershoot_perFPM_allclean = \
            corr_matrix_median = corr_matrix_average = indices_good_det = pd.DataFrame()
        command = 'mv ' + suffix + '*.png ' + suffix + '*.txt *.log ./' + outputdir + '/' + directory_specific_cuts
        os.system(command)
        return (merged_median_undershoot_perFPM_allclean, merged_average_undershoot_perFPM_allclean,
                corr_matrix_median, corr_matrix_average, indices_good_det)

    # remove N "largestfpms_to_flag" detectors with largest undershoot
    max_indices_all = merged_average_undershoot_perFPM_nonan.apply(lambda col: col.nlargest(
        largestfpms_to_flag).index.tolist())
    max_indices_all_flat_unique = np.unique(max_indices_all.to_numpy().flatten())
    max_indices = pd.Index(max_indices_all_flat_unique)
    logger.info('{} detectors removed showing largest undershoots, these are {}'.format(largestfpms_to_flag,
                                                                                        max_indices.to_list()))
    merged_average_undershoot_perFPM_maxnanremoved = merged_average_undershoot_perFPM_nonan.drop(max_indices)

    if merged_average_undershoot_perFPM_maxnanremoved.empty:
        logger.info('Removing {} detectors with largest undershoot each month resulted in the excision of all '
                    'detectors - returning empty dataframes'.format(largestfpms_to_flag))
        merged_median_undershoot_perFPM_allclean = merged_average_undershoot_perFPM_allclean = \
            corr_matrix_median = corr_matrix_average = indices_good_det = pd.DataFrame()
        command = 'mv ' + suffix + '*.png ' + suffix + '*.txt *.log ./' + outputdir + '/' + directory_specific_cuts
        os.system(command)
        return (merged_median_undershoot_perFPM_allclean, merged_average_undershoot_perFPM_allclean,
                corr_matrix_median, corr_matrix_average, indices_good_det)

    # Drop detector 63 which is consistently larger - It may already be removed by flags above
    index_to_remove = '63'
    logger.info('Detector 63 always removed')
    if index_to_remove in merged_average_undershoot_perFPM_maxnanremoved.index:
        merged_average_undershoot_perFPM_maxnanremoved = (
            merged_average_undershoot_perFPM_maxnanremoved.drop(index_to_remove))

    # Separate merged dataframe into average and median (and stdv)
    merged_average_undershoot_perFPM_maxnanremoved_dropstdv = merged_average_undershoot_perFPM_maxnanremoved[
        merged_average_undershoot_perFPM_maxnanremoved.columns.drop(
            list(merged_average_undershoot_perFPM_maxnanremoved.filter(regex='stdv')))]
    # Get median out
    merged_median_undershoot_perFPM_allclean = merged_average_undershoot_perFPM_maxnanremoved_dropstdv[
        merged_average_undershoot_perFPM_maxnanremoved_dropstdv.columns.drop(
            list(merged_average_undershoot_perFPM_maxnanremoved_dropstdv.filter(regex='average')))]
    # Get average out
    merged_average_undershoot_perFPM_allclean = merged_average_undershoot_perFPM_maxnanremoved_dropstdv[
        merged_average_undershoot_perFPM_maxnanremoved_dropstdv.columns.drop(
            list(merged_average_undershoot_perFPM_maxnanremoved_dropstdv.filter(regex='median')))]

    # Create correlation matrix and pair plot for each of the above
    corr_matrix_median = plotcorrandpair(merged_median_undershoot_perFPM_allclean, outputfile=suffix + '_median')
    corr_matrix_average = plotcorrandpair(merged_average_undershoot_perFPM_allclean, outputfile=suffix + '_average')

    # Writing final pandas dataframe to .csv file
    MkfFileOps(merged_average_undershoot_perFPM_maxnanremoved).write_mkf_to_csv(suffix + '_clean_averageinfo')

    # Get indices (detectors) and save them to csv file
    indices_good_det = pd.DataFrame(merged_average_undershoot_perFPM_allclean.index,
                                    columns=['Det_number']).set_index('Det_number', drop=False)
    indices_good_det.set_index('Det_number', drop=False, inplace=True)
    indices_good_det.index.names = ['Index']
    logger.info('Good detectors {}'.format(indices_good_det['Det_number'].to_list()))
    # Save the index DataFrame to a CSV file
    MkfFileOps(indices_good_det).write_mkf_to_csv(suffix + '_accepteddetectors', saveindex=False)

    # Create a text file of all detectors with bad ones set to NaN
    # creating dataframe of all detectors and cleaning up indices to match detector number
    nicDET_geograph_df = (pd.DataFrame(nicDET_geograph, columns=['Det_number'])).set_index('Det_number', drop=False)
    nicDET_geograph_df.index.names = ['Index']
    # Merging the two above and adding NaN to the ones that are missing from indices_good_det
    full_good_bad_detector = pd.merge(indices_good_det, nicDET_geograph_df, how='outer', left_index=True,
                                      right_index=True).drop(columns=['Det_number_y'])
    full_good_bad_detector.rename(columns={'Det_number_x': 'Det_number'}, inplace=True)
    # Creating a CSV of this dataframe
    MkfFileOps(full_good_bad_detector).write_mkf_to_csv(suffix + '_detectorstatus', saveindex=False)

    # Moving files to proper directory
    command = 'mv ' + suffix + '*.png ' + suffix + '*.txt *.log ./' + outputdir + '/' + directory_specific_cuts
    os.system(command)

    # Visualize and create a list of on/off detectors
    # to-do

    return (merged_median_undershoot_perFPM_allclean, merged_average_undershoot_perFPM_allclean,
            corr_matrix_median, corr_matrix_average, indices_good_det)


def readmkfsfromtextfile(mkfsintextfile):
    with open(mkfsintextfile) as file:
        listofmkffiles = [line.rstrip() for line in file]
    return listofmkffiles


def plotcorrandpair(mkf_averages, outputfile):
    corr_matrix = mkf_averages.corr()

    # Plot the correlation matrix using seaborn
    ###########################################
    plt.figure(figsize=(18, 16))
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=0, vmax=1, annot=True, square=True,
                linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix')
    # Save the plot
    plt.tight_layout()
    plotname = outputfile + '_corrmatrix.png'
    plt.savefig(plotname, format='png', dpi=200)
    plt.close()

    # Plotting the pair plots
    #########################
    sns.set_context("paper", font_scale=2.5)
    sns.pairplot(mkf_averages, corner=True, diag_kind='kde')
    # Save the plot
    plt.tight_layout()
    plotname = outputfile + '_pairplot.png'
    plt.savefig(plotname, format='png', dpi=200)
    plt.close()

    # Plot median per detector as a function of time
    ################################################
    fig, ax1 = plt.subplots(1, figsize=(34, 16), dpi=80, facecolor='w', edgecolor='k')
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_xlabel('Date (2023 July - 2024 December)', fontsize=18)
    ax1.set_ylabel('Median undershoot per month',
                   fontsize=18)
    ax1.xaxis.offsetText.set_fontsize(18)

    for index, row in mkf_averages.iterrows():
        ax1.plot(row, ls='', marker='o', markersize=14, label=index, alpha=0.5, )

    ax1Leg = ax1.legend(loc='lower right', fontsize=18, bbox_to_anchor=(1.13, 0), framealpha=None, ncol=3)
    ax1Leg.get_frame().set_linewidth(1)
    ax1Leg.get_frame().set_edgecolor('k')

    # Saving figure
    plotname = outputfile + 'underperfpm_vs_time.png'
    fig.savefig(plotname, format='png', dpi=200)
    plt.close()

    return corr_matrix


def main():
    parser = argparse.ArgumentParser(description="Compare undershoots in a list of mkf files")
    parser.add_argument("mkffiles", help=".txt list of NICER MKF files", type=str)
    parser.add_argument("-ss", "--sunshine", help="Filtering for sunshine, 0 for night, 1 for day, "
                                                  "and 2 for no filtering (default=1)", type=int, default=2)
    parser.add_argument("-sl", "--sunAngLR", help="Filtering for sun angle, lower-range", type=float,
                        default=45)
    parser.add_argument("-su", "--sunAngUR", help="Filtering for sun angle, upper-range", type=float,
                        default=180)

    parser.add_argument("-ml", "--moonAngLR", help="Filtering for moon angle, lower-range", type=float,
                        default=0)
    parser.add_argument("-mu", "--moonAngUR", help="Filtering for moon angle, upper-range", type=float,
                        default=180)
    parser.add_argument("-mpl", "--moonphaseLR", help="Filtering for moon phase, lower-range", type=float,
                        default=0)
    parser.add_argument("-mpu", "--moonphaseUR", help="Filtering for moon phase, upper-range", type=float,
                        default=1)
    parser.add_argument("-bel", "--brearthLR", help="Filtering for bright Earth angle, lower-range",
                        type=float, default=0)
    parser.add_argument("-beu", "--brearthUR", help="Filtering for bright Earth angle, upper-range",
                        type=float, default=180)
    parser.add_argument("-al", "--sunAzLR", help="Filtering for sun azimuth (clocking) angle, "
                                                 "lower-range", type=float,
                        default=-180)
    parser.add_argument("-au", "--sunAzUR", help="Filtering for sun azimuth (clocking) angle, "
                                                 "upper-range", type=float,
                        default=180)
    parser.add_argument("-tf", "--timepostleak", help="Accept only time post repair (default=True)",
                        type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("-ud", "--under", help="Which under parameter to consider (MPU_UNDERONLY_COUNT, "
                                               "MPU_UNDER_COUNT) ", type=str, default='MPU_UNDERONLY_COUNT')
    parser.add_argument("-lff", "--largestfpms_to_flag", help="Flag N FPMs with largest under shoot"
                                                              "(default=0)", type=int, default=0)
    parser.add_argument("-dp", "--diagnosticsplots", help="Create all diagnostics plot (default=False)",
                        type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-od", "--outputdir", help="Name of output directory", type=str,
                        default='compare_mkfs')
    args = parser.parse_args()

    # Reading mkf .txt file
    listofmkffiles = readmkfsfromtextfile(args.mkffiles)
    # Running primary function
    comparemkfundershoots(listofmkffiles, args.sunshine, args.sunAngLR, args.sunAngUR, args.moonAngLR, args.moonAngUR,
                          args.moonphaseLR, args.moonphaseUR, args.brearthLR, args.brearthUR, args.sunAzLR,
                          args.sunAzUR, args.timepostleak, args.under, args.largestfpms_to_flag, args.diagnosticsplots,
                          args.outputdir)


if __name__ == '__main__':
    main()
