'''
The purpose of this script is to convert FDR tiff's into connectivity rasters
'''

import argparse, os, sys



def create_pytopkapi_hdf5_from_nc(nc_f, mask_tiff, output_folder=""):
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

    # import shutil
    # shutil.copy2('/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/rainfields.h5', output_folder)
    # shutil.copy2('/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/ET.h5', output_folder)

    # with h5py.File(rainfall_outputFile, 'w') as f2:
    #     # f2.create_group(u'sample_event')
    #     # f2[u'sample_event'].create_dataset(u'rainfall', shape=(time_step, no_of_cell), dtype='f')
    #     # grp0 = f2.create_group('test')
    #     print ('H5py description', str(h5py))
    #     print ('H5py create_group description', str(f2.create_group))
    #
    #
    #     daataa = numpy.zeros((time_step, no_of_cell))
    #     for i in range(time_step):
    #         ppt_at_that_time_step = ppt[i]
    #         daataa[i, :] = ppt_at_that_time_step[mask == 1]
    #
    #     group_name = 'sample_event/rainfall'.encode('utf-8')
    #     dset = f2.create_dataset(group_name, data= daataa)
    #     # dset = f2.create_dataset("sample_event/rainfall", data= daataa, dtype="S10")

    with h5py.File(rainfall_outputFile, 'w') as f2:
        # f2.create_group(u'sample_event')
        # f2[u'sample_event'].create_dataset(u'rainfall', shape=(time_step, no_of_cell), dtype='f')
        # grp0 = f2.create_group('test')
        print ('H5py description', str( h5py))
        print ('H5py create_group description', str(f2.create_group))

        # grp = f2.create_group('sample_event')  #.encode('utf-8')
        group_name = 'sample_event/rainfall'.encode('utf-8')
        f2.create_dataset( group_name, shape=(time_step, no_of_cell), dtype='f')

        rainArray = f2[u'sample_event'][u'rainfall']

        data = numpy.zeros((time_step, no_of_cell))
        for i in range(time_step):
            ppt_at_that_time_step = ppt[i]
            data[i, :] = ppt_at_that_time_step[mask==1]

        rainArray[:] = data

    # :TODO: Change the empty ET to calculated
    with h5py.File(ET_outputFile, 'w') as f1:
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

    print ('Progress--> Rainfall file successfully created. Shape (%s x %s)',(time_step, no_of_cell))
    return rainfall_outputFile

def create_pytopkapi_hdf5_from_nc_unit_mm_per_timestep(nc_f, mask_tiff, timestep_in_hr=24, output_folder=""):
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
        ppt_at_that_time_step = ppt[i]                                                  # the unit is m/hr
        ppt_at_that_time_step = ppt_at_that_time_step * int(timestep_in_hr) * 1000.0    # to convert into mm/timestep
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
    parser.add_argument('-i', '--input_ppt', required=True, help="input NetCDF ppt file")
    parser.add_argument('-m', '--input_mask', required=False, help="input mask GeoTIFF file")
    parser.add_argument('-o', '--output_folder', required=True, help="output h5py ppt file")
    parser.add_argument('-t', '--timestep', required=True, help="timestep used in the model ")


    args = parser.parse_args()

    if not os.path.exists(args.input_ppt):
        print ('Could not find input input ppt, please make sure the path is correct at %s', args.input_ppt)
        sys.exit(1)

    arg1 = args.input_ppt
    arg2 = args.input_mask
    arg3 = args.output_folder
    arg4 = args.timestep

    # Default, for python3
    # create_pytopkapi_hdf5_from_nc(nc_f=arg1, mask_tiff=arg2 ,  output_folder=arg3)
    create_pytopkapi_hdf5_from_nc_unit_mm_per_timestep(nc_f=arg1, mask_tiff=arg2 ,  output_folder=arg3, timestep_in_hr=arg4)




