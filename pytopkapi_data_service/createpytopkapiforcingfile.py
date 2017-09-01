'''
The purpose of this script is to convert FDR tiff's into connectivity rasters
'''

import argparse, os, sys, numpy

def change_timestep(ppt_ar, new_timestep_hr, old_timestep_hr=24):
    """
    :param ppt_ar:           Its is a 3d numpy array. e.g. ppt_ar.shape = (365,786,1008)
    :param new_timestep_hr:
    :param old_timestep_hr:
    :return:
    """
    #:todo Change hard coding timestep and source information
    print ("Progress >> Trying to match timestep. Old timestep: ",old_timestep_hr)

    timestep_factor = int(old_timestep_hr/ int(new_timestep_hr))  # time interval should divide 24 without leaving any remainders
    new_ppt = numpy.zeros((int(ppt_ar.shape[0]) / timestep_factor, int(ppt_ar.shape[1]), int(ppt_ar.shape[2])))

    for i in range(0, len(ppt_ar), timestep_factor):
        k = 1
        sum = numpy.zeros((int(ppt_ar.shape[1]), int(ppt_ar.shape[2])))
        for j in range(timestep_factor):
            sum = sum + ppt_ar[i + j]
            k = k + 1
            new_ppt[i / timestep_factor] = sum / k
    print ("Progress >> Trying to match timestep. New timestep: ", new_timestep_hr)
    return  new_ppt


def create_pytopkapi_hdf5_from_nc_unit_mm_per_timestep(nc_rain, nc_et, mask_tiff, timestep_in_hr=24, output_folder="", source='daymet'): #daymet, ueb
    import h5py, numpy
    from netCDF4 import Dataset
    from osgeo import gdal

    # read mask
    dset = gdal.Open(mask_tiff)
    mask = dset.ReadAsArray()                   # a (x*y) array of values from raster



    # read netCDF rain
    root = Dataset(nc_rain, 'r')

    if source =='ueb':
        ppt = root.variables['SWIT'][:] * 1000.0    # all the precipitation records, in 3d array (time * x * y)
        ppt = numpy.flip(ppt, axis=1)               # UEB array is mirrored in y axis
        print ('mask shape: %s, Original nc_rain_shape %s' % (mask.shape, ppt.shape))
        # ppt[ppt<0.001]=0
        ppt = change_timestep(ppt_ar=ppt, new_timestep_hr=6, old_timestep_hr=24)
        print ('Progress --> PPT: Max, Min, and mean value of ppt: ', ppt.max(), ppt.min(), ppt.mean())
    else:  #elif source=='daymet':
        ppt = root.variables['prcp'][:]        # all the precipitation records, in 3d array (time * x * y)

    time_step = ppt.shape[0]                    # time length of the rainfall 3d array
    no_of_cell = mask[mask == 1].size           # mask[mask==1] creates a 1d array satisfying condition mask==1
    print ('Progress -->> PPT: file read: ',nc_rain, ' Its dimension is ', ppt.shape)

    # output path
    rainfall_outputFile = os.path.join(output_folder, "rainfields.h5")
    ET_outputFile = os.path.join(output_folder, "ET.h5")


    f2 =  h5py.File(rainfall_outputFile, 'w')
    group_name = 'sample_event/rainfall'.encode('utf-8')
    f2.create_dataset( group_name, shape=(time_step, no_of_cell), dtype='f') ; print ('Progress>> In hdf5 file, dataset created .')

    rainArray = f2[u'sample_event'][u'rainfall']

    data = numpy.zeros((time_step, no_of_cell))
    for i in range(time_step):
        ppt_at_that_time_step = ppt[i]                                                  # the unit is mm/day
        # ppt_at_that_time_step = ppt_at_that_time_step * int(timestep_in_hr)/24.0        # to convert into mm/timestep
        data[i, :] = ppt_at_that_time_step[mask==1]

    rainArray[:] = data
    f2.close()

    print ('Progress --> PPT wrote..')



    # # # ET
    root = Dataset(nc_et, 'r')

    if source =='ueb':
        et_ar = ppt * 0  # all the precipitation records, in 3d array (time * x * y)
        print ('Progress --> ET: Max, Min, and mean value of ppt: ', et_ar.max(), et_ar.min(), et_ar.mean())
    else: #elif  source == 'daymet':
        et_ar = root.variables['ETr'][:]


    f1 =  h5py.File(ET_outputFile, 'w')
    f1.create_group('sample_event')
    f1['sample_event'].create_dataset('ETo', shape=(time_step, no_of_cell), dtype='f')
    f1['sample_event'].create_dataset('ETr', shape=(time_step, no_of_cell), dtype='f')

    EToArray = f1['sample_event']['ETo']
    ETrArray = f1['sample_event']['ETr']

    data = numpy.zeros((time_step, no_of_cell))
    for i in range(time_step):
        data[i, :] = et_ar[i][mask==1]

    ETrArray[:] = data
    EToArray[:] = data
    f1.close()

    print ('Progress--> Rainfall / ET file successfully created. Shape (%s x %s)'%(time_step, no_of_cell))

    # try:
    #     os.remove(nc_rain) # delete the NetCDF file
    #     os.remove(nc_et)  # delete the NetCDF file
    # except:
    #     pass
    # print ('Progress --> Forcing files created')
    return rainfall_outputFile, ET_outputFile


def create_pytopkapi_hdf5_from_nc_workingbackup(nc_f, mask_tiff, output_folder=""):
    import h5py, numpy
    from netCDF4 import Dataset
    from osgeo import gdal

    root = Dataset(nc_f, 'r')
    ppt = root.variables['prcp'][:]             # all the precipitation records, in 3d array (time * x * y)

    dset = gdal.Open(mask_tiff)
    mask = dset.ReadAsArray()                   # a (x*y) array of values from raster

    time_step = ppt.shape[0]                    # time length of the rainfall 3d array
    no_of_cell = mask[mask == 1].size          # mask[mask==1] creates a 1d array satisfying condition mask==1

    # output path
    rainfall_outputFile = os.path.join(output_folder, "rainfields.h5")
    ET_outputFile = os.path.join(output_folder, "ET.h5")


    f2 =  h5py.File(rainfall_outputFile, 'w')
    print ('H5py description', str( h5py))
    print ('H5py create_group description', str(f2.create_group))

    # grp = f2.create_group('sample_event')  #.encode('utf-8')
    group_name = 'sample_event/rainfall'.encode('utf-8')
    f2.create_dataset( group_name, shape=(time_step, no_of_cell), dtype='f')

    rainArray = f2[u'sample_event'][u'rainfall']

    data = numpy.zeros((time_step, no_of_cell))
    for i in range(time_step):
        ppt_at_that_time_step = ppt[i]                                  # i think the unit is meters/hr
        data[i, :] = ppt_at_that_time_step[mask==1]

    rainArray[:] = data
    f2.close()


    # :TODO: Change the empty ET to calculated
    f1 =  h5py.File(ET_outputFile, 'w')
    f1.create_group('sample_event')
    f1['sample_event'].create_dataset('ETo', shape=(time_step, no_of_cell), dtype='f')
    f1['sample_event'].create_dataset('ETr', shape=(time_step, no_of_cell), dtype='f')

    EToArray = f1['sample_event']['ETo']
    ETrArray = f1['sample_event']['ETr']

    data = numpy.zeros((time_step, no_of_cell))
    for i in range(time_step):
        data[i, :] = numpy.random.rand(1, no_of_cell) * 0.0

    EToArray = data
    ETrArray = data
    f1.close()

    print ('Progress--> Rainfall file successfully created. Shape (%s x %s)'%(time_step, no_of_cell))

    try:
        os.remove(nc_f) # delete the NetCDF file
    except:
        pass

    return rainfall_outputFile


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ippt', '--input_ppt', required=True, help="input NetCDF ppt file")
    parser.add_argument('-iet', '--input_et', required=True, help="input NetCDF ppt file")
    parser.add_argument('-m', '--input_mask', required=False, help="input mask GeoTIFF file")
    parser.add_argument('-o', '--output_folder', required=True, help="output h5py ppt file")
    parser.add_argument('-t', '--timestep', required=True, help="timestep used in the model ")


    args = parser.parse_args()

    if not os.path.exists(args.input_ppt):
        print ('Could not find input input ppt, please make sure the path is correct at %s', args.input_ppt)
        sys.exit(1)

    arg1 = args.input_ppt
    arg2 = args.input_et
    arg3 = args.input_mask
    arg4 = args.output_folder
    arg5 = args.timestep

    # Default, for python3
    # create_pytopkapi_hdf5_from_nc(nc_f=arg1, mask_tiff=arg2 ,  output_folder=arg3)
    create_pytopkapi_hdf5_from_nc_unit_mm_per_timestep(nc_rain=arg1, nc_et=arg2, mask_tiff=arg3 ,
                                                       output_folder=arg4, timestep_in_hr=arg5)




