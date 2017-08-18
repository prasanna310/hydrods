import numpy as np
import pandas as pd
import argparse
from shutil import copyfile

import shlex
import subprocess
import logging

logger = logging.getLogger(__name__)

def call_subprocess(cmdString=None, debugString=None):
    cmdargs = shlex.split(cmdString)
    # debFile = open('debug_file.txt', 'w')
    # debFile.write('Starting %s \n' % debugString)
    errorString = "Error in " + debugString + ". The message returned from the application is: "
    retValue = subprocess.call(cmdargs, stdout=None)
    if (retValue==0):
        # debFile.write('%s Successful\n' % debugString)
        # debFile.close()
        logger.info("subprocess success." + debugString + ". Return value from the application is: " + str(retValue))
        retDictionary = {'success': "True", 'message': debugString + ". Return value from the application is: "
                                                       + str(retValue)}
    else:
        # debFile.write('There was error in %s\n' % debugString)
        # debFile.close()
        logger.error("subprocess failed." + debugString + ". Return value from the application is: " + str(retValue))
        retDictionary = {'success': "False", 'message': errorString + str(retValue)}

    # print('call_subprocess retDictionary:')
    # print(retDictionary)
    # print('\n')
    # print('call_subprocess retValue:' + str(retValue))
    return retDictionary


def reclassify_raster_with_LUT( input_raster,  LUT, output_raster='reclassified_raster.tif',  delimiter=","):


    # LUT_array = np.genfromtxt(LUT, delimiter=delimiter)
    LUT_array = np.array ( pd.read_csv(LUT)  )

    copyfile(input_raster, 'temp.tif')

    calc_list = ['%s*(A==%s)' % (new, old) for old, new in LUT_array]
    calc_string = "+".join(calc_list)
    cmdString = 'gdal_calc.py -A %s --outfile=%s --calc="'%('temp.tif', output_raster) + calc_string + '" --type=Float32'
    return call_subprocess(cmdString, 'reclassify raster')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_raster', required=True, help='Input raster')
    parser.add_argument('-lut', '--lookuptable', required=True, help='lookup table')
    parser.add_argument('-o', '--output_raster', required=False, help='Name of output raster file')
    parser.add_argument('-sep', '--seperator', required=False, help='seperator used in the look up table file. e.g. , ')
    args = parser.parse_args()

    reclassify_raster_with_LUT(input_raster=args.input_raster, LUT=args.lookuptable, output_raster=args.output_raster, delimiter=args.seperator)

