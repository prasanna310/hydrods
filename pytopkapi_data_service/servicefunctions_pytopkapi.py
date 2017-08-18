import numpy as np
import os, sys, zipfile, h5py, json, shutil
import configparser
from configparser import SafeConfigParser
from usu_data_service.servicefunctions.utils import call_subprocess
from usu_data_service.utils import generate_uuid_file_path, delete_working_uuid_directory
from usu_data_service.servicefunctions.watershedFunctions import create_OutletShape
from usu_data_service.servicefunctions.terrainFunctions import uncompressRaster, project_shapefile_EPSG
from datetime import datetime
from usu_data_service.servicefunctions.terrainFunctions import *
from usu_data_service.servicefunctions.watershedFunctions import *
from usu_data_service.servicefunctions.canopyFunctions import *
from usu_data_service.servicefunctions.netcdfFunctions import *
from usu_data_service.topnet_data_service.TOPNET_Function.CommonLib import download_soil_data_for_pytopkapi
import urllib
import pandas as pd

import pytopkapi
from pytopkapi.results_analysis import plot_Qsim_Qobs_Rain, plot_soil_moisture_maps
import pytopkapi.utils as ut

from datetime import datetime, timedelta
from osgeo import gdal, ogr


def get_cellSize(tif_file):
    try:

        from osgeo import gdal
        dset = gdal.Open(tif_file)

        x0, dx, fy, y0, fx, dy = dset.GetGeoTransform()
        print ('Progress --> Cell size calculated is %s m' % dx)
    except:
        dx = ""
        print ("Either no GDAL, or no tiff file")
    return dx


def get_raster_detail(input_raster, output_json=None):
    minx = None
    miny = None
    maxx = None
    maxy = None
    dx = ncol = nrow = bands = None

    try:
        temp_raster = 'temp.tif'
        os.system('gdalwarp -t_srs "+proj=longlat +ellps=WGS84" %s %s -overwrite' % (input_raster, temp_raster))

        ds = gdal.Open(temp_raster)
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + width * gt[4] + height * gt[5]
        maxx = gt[0] + width * gt[1] + height * gt[2]
        maxy = gt[3]

        x0, dx, fy, y0, fx, dy = ds.GetGeoTransform()
        ncol = ds.RasterXSize
        nrow = ds.RasterYSize
        bands = ds.RasterCount

    except Exception as e:
        print (' Progres --> Error: Tiff files contents is not supported. Try another Tiff file', e)
        # return {'success': 'True', 'message':e}

    if not output_json == None:
        JSON_dict = {}
        JSON_dict['minx'] = minx
        JSON_dict['miny'] = miny
        JSON_dict['maxx'] = maxx
        JSON_dict['maxy'] = maxy

        JSON_dict['dx'] = dx
        JSON_dict['cell_size'] = dx
        JSON_dict['ncol'] = ncol
        JSON_dict['nrow'] = nrow
        JSON_dict['bands'] = bands

        # save
        with open(output_json, 'w') as newfile:
            json.dump(JSON_dict, newfile)

    print (' minx, maxx, miny, maxy', minx, maxx, miny, maxy)
    return {'success': 'True', 'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy, 'cell_size': dx, 'dx': dx,
            'ncol': ncol, 'nrow': nrow, 'bands': bands}


def run_model_with_input_as_dictionary(inputs_dictionary, topkapi_ini_fname="TOPKAPI.ini"):
    """
    :param inputs_dictionary:   inputs converted to dictionary in validation step. Type of inouts are taken care of
                                e.g. float is already a float type, int is int, and string is string.
    :param path_to_project_ini:	HydroShare location of the inputs?
    :return:                    Hydrograph (as timeseries, between datetime Vs Discharge)
    """
    # inputs extracted from the dictionary

    user_name = inputs_dictionary['user_name']
    simulation_name = inputs_dictionary['simulation_name']
    simulation_folder = inputs_dictionary['simulation_folder']  # or, = path_to_project_ini
    simulation_start_date = inputs_dictionary['simulation_start_date']
    simulation_end_date = inputs_dictionary['simulation_end_date']
    USGS_gage = int(inputs_dictionary['USGS_gage'])

    outlet_x = float(inputs_dictionary['outlet_x'])
    outlet_y = float(inputs_dictionary['outlet_y'])
    box_topY = float(inputs_dictionary['box_topY'])
    box_bottomY = float(inputs_dictionary['box_bottomY'])
    box_rightX = float(inputs_dictionary['box_rightX'])
    box_leftX = float(inputs_dictionary['box_leftX'])

    timeseries_source = inputs_dictionary['timeseries_source']
    threshold = float(inputs_dictionary['box_bottomY'])
    cell_size = float(inputs_dictionary['box_bottomY'])
    timestep = float(inputs_dictionary['box_bottomY'])
    model_engine = inputs_dictionary['model_engine']

    ini_path = os.path.join(simulation_folder, topkapi_ini_fname)

    # TOPKAPI MODEL
    if model_engine == 'TOPKAPI':
        # step0,
        run_1 = pytopkapi_run_instance(simulation_name=simulation_name, cell_size=cell_size, timestep=timestep,
                                       xy_outlet=[outlet_x, outlet_y],
                                       yyxx_boundingBox=[box_topY, box_bottomY, box_leftX, box_rightX],
                                       USGS_gage=USGS_gage, list_of_threshold=[threshold],
                                       simulation_folder=simulation_folder)

        step1_create_ini = run_1.prepare_supporting_ini()  # step1
        # step2_run_model = run_1.run()                             # step2
        date_in_datetime, Qsim, error = run_1.get_Qsim_and_error()

        # if write_to_db:
        # # write_to_db_input_as_dictionary(inputs_dictionary, simulation_folder)
        # table_id = write_to_db_input_as_dictionary(inputs_dictionary, simulation_folder)

        # create_viewplot_hydrograph(date_in_datetime, Qsim, error)  # aile kina ho kaam garena

        # preparing timeseries data in the format shown in: http://docs.tethysplatform.org/en/latest/tethys_sdk/gizmos/plot_view.html#time-series
        hydrograph_series = []
        date_broken = [[dt.year, dt.month, dt.day, dt.hour, dt.minute] for dt in date_in_datetime]
        for i in range(len(Qsim)):
            date = datetime(year=date_broken[i][0], month=date_broken[i][1], day=date_broken[i][2],
                            hour=date_broken[i][3],
                            minute=date_broken[i][4])
            hydrograph_series.append([date, float(Qsim[i])])

    return hydrograph_series


def create_config_files_create_file(path_to_in_cell_param_dat, path_to_out_ini, raster_files_dict,
                                    numeric_param={'pvs_t0': 40., 'vo_t0': 1000.0, 'qc_t0': 0, 'kc': 1.0}):
    '''
    raster_files_dict: 		Input, A dictionary that has path to the required geospatial/soil tif files
    numeric_param_dict: 	Input, A dictionary that has values of the numeric parameters
    create_file.ini:		Output, the config files that stores these information.
                            This name is NOT NECESSARILY the output. The hydroDS has its own system to create output name
    '''

    configWrite = configparser.RawConfigParser()
    configWrite.add_section('raster_files')
    configWrite.set('raster_files', 'dem_fname', raster_files_dict['dem_fname'])
    configWrite.set('raster_files', 'mask_fname', raster_files_dict['mask_fname'])
    configWrite.set('raster_files', 'soil_depth_fname', raster_files_dict['soil_depth_fname'])
    configWrite.set('raster_files', 'conductivity_fname', raster_files_dict['conductivity_fname'])
    configWrite.set('raster_files', 'hillslope_fname', raster_files_dict['hillslope_fname'])
    configWrite.set('raster_files', 'sat_moisture_content_fname', raster_files_dict['sat_moisture_content_fname'])
    configWrite.set('raster_files', 'resid_moisture_content_fname', raster_files_dict['resid_moisture_content_fname'])
    configWrite.set('raster_files', 'bubbling_pressure_fname', raster_files_dict['bubbling_pressure_fname'])
    configWrite.set('raster_files', 'pore_size_dist_fname', raster_files_dict['pore_size_dist_fname'])
    configWrite.set('raster_files', 'overland_manning_fname', raster_files_dict['overland_manning_fname'])
    configWrite.set('raster_files', 'channel_network_fname', raster_files_dict['channel_network_fname'])
    configWrite.set('raster_files', 'flowdir_fname', raster_files_dict['flowdir_fname'])
    # configWrite.set('raster_files', 'channel_manning_fname', raster_files_dict['channel_manning_fname'])
    configWrite.set('raster_files', 'flowdir_source', 'TauDEM')

    configWrite.add_section('output')
    configWrite.set('output', 'param_fname', path_to_in_cell_param_dat)

    configWrite.add_section('numerical_values')
    configWrite.set('numerical_values', 'pvs_t0', numeric_param['pvs_t0'])
    configWrite.set('numerical_values', 'vo_t0', numeric_param['vo_t0'])
    configWrite.set('numerical_values', 'qc_t0', numeric_param['qc_t0'])
    configWrite.set('numerical_values', 'kc', numeric_param['kc'])

    with open(path_to_out_ini, 'wt') as configFile:
        configWrite.write(configFile)

    return os.path.join(path_to_out_ini)


def create_config_files_zero_slope_mngmt(path_to_unmodified_cell_param_dat, path_to_modified_cell_param_dat,
                                         path_to_out_ini, cell_size=30.92208078):
    configWrite = configparser.RawConfigParser()
    configWrite.add_section('input_files')
    configWrite.set('input_files', 'file_cell_param', path_to_unmodified_cell_param_dat)

    configWrite.add_section('output_files')
    configWrite.set('output_files', 'file_cell_param_out',
                    path_to_modified_cell_param_dat)  # in HydroDS, we dont need to give absolute path

    configWrite.add_section('numerical_values')
    configWrite.set('numerical_values', 'nb_param', int(21))
    configWrite.set('numerical_values', 'X', cell_size)

    with open(path_to_out_ini, 'wt') as configFile:
        configWrite.write(configFile)

    return path_to_out_ini


def create_config_files_plot_flow_precip(path_to_out_ini,
                                         files={'path_to_results': 'results.h5', 'path_to_rain': 'rainfields.h5',
                                                'path_to_runoff_file': 'runoff.dat'},
                                         parameters={'outlet_ID': '', 'calibration_start_date': ''}):
    configWrite = configparser.RawConfigParser()
    configWrite.add_section('files')
    configWrite.set('files', 'file_Qsim', files['path_to_results'])
    configWrite.set('files', 'file_Qobs', files['path_to_runoff_file'])
    configWrite.set('files', 'file_rain', files['path_to_rain'])
    configWrite.set('files', 'image_out', '')

    configWrite.add_section('groups')
    configWrite.set('groups', 'group_name', 'sample_event')

    configWrite.add_section('parameters')
    configWrite.set('parameters', 'outlet_ID', parameters['outlet_ID'])
    configWrite.set('parameters', 'graph_format', 'png')
    configWrite.set('parameters', 'start_calibration', parameters['calibration_start_date'] + " ;dd/mm/yyyy")

    configWrite.add_section('flags')
    configWrite.set('flags', 'Qobs', 'True')
    configWrite.set('flags', 'Pobs', 'True')
    configWrite.set('flags', 'nash', 'True')
    configWrite.set('flags', 'R2', 'True')
    configWrite.set('flags', 'RMSE', 'True')
    configWrite.set('flags', 'RMSE_norm', 'True')
    configWrite.set('flags', 'Diff_cumul', 'True')
    configWrite.set('flags', 'Bias_cumul', 'True')
    configWrite.set('flags', 'Err_cumul', 'True')
    configWrite.set('flags', 'Abs_cumul', 'True')

    with open(path_to_out_ini, 'wt') as configFile:
        configWrite.write(configFile)

    return path_to_out_ini


def create_config_files_plot_soil_moisture_map(path_to_out_ini,
                                               files={'file_global_param': 'global_param.dat',
                                                      'file_cell_param': 'cell_param.dat', 'results': 'results.h5'},
                                               paramters={},
                                               calib_param={'fac_l': 1., 'fac_ks': 1., 'fac_n_o': 1., 'fac_n_c': 1.},
                                               flags={'t1': 1, 't2': 5, 'variable': 4}):
    configWrite = configparser.RawConfigParser()
    configWrite.add_section('files')
    configWrite.set('files', 'file_global_param', files['file_global_param'])
    configWrite.set('files', 'file_cell_param', files['file_cell_param'])
    configWrite.set('files', 'file_sim', files['results'])

    configWrite.add_section('paths')
    configWrite.set('paths', 'path_out', files['soil_maps_folder'])

    configWrite.add_section('calib_params')
    configWrite.set('calib_params', 'fac_l', calib_param['fac_l'])
    configWrite.set('calib_params', 'fac_ks', calib_param['fac_ks'])
    configWrite.set('calib_params', 'fac_n_o', calib_param['fac_n_o'])
    configWrite.set('calib_params', 'fac_n_c', calib_param['fac_n_c'])

    configWrite.add_section('flags')
    configWrite.set('flags', 't1', paramters['start_timestep'])
    configWrite.set('flags', 't2', paramters['end_timestep'])
    configWrite.set('flags', 'variable', paramters['variable_to_plot_maps'])

    with open(path_to_out_ini, 'wt') as configFile:
        configWrite.write(configFile)

    return path_to_out_ini


def create_config_files_TOPKAPI_ini(path_to_out_ini,
                                    input_files={},
                                    output_files={},
                                    calib_param={'fac_l': 1., 'fac_ks': 1., 'fac_n_o': 1., 'fac_n_c': 1.,
                                                 'fac_th_s': 1},
                                    numerical_options={'solve_s': 1, 'solve_o': 1, 'solve_c': 1,
                                                       'only_channel_output': 'False'},
                                    external_flow={'external_flow_status': 'False'},
                                    # TODO: This needs external flow information if it exists
                                    ):
    configWrite = configparser.RawConfigParser()
    # configWrite.add_section('input_files')      # :TODO HARD CODED RIGHT NOW. See what the problem is and change this
    # configWrite.set('input_files', 'file_global_param', os.path.join(simulation_folder,'global_param.dat'))
    # configWrite.set('input_files', 'file_cell_param', os.path.join(simulation_folder,'cell_param.dat'))
    # configWrite.set('input_files', 'file_rain', os.path.join(simulation_folder,'rainfields.h5'))
    # configWrite.set('input_files', 'file_et',  os.path.join(simulation_folder,'ET.h5'))
    #
    # configWrite.add_section('output_files')
    # configWrite.set('output_files', 'file_out', os.path.join(simulation_folder,'results.h5'))
    # configWrite.set('output_files', 'file_change_log_out', os.path.join(simulation_folder,'change_result_log.dat'))
    # configWrite.set('output_files', 'append_output', 'False')


    configWrite.add_section('input_files')
    configWrite.set('input_files', 'file_global_param', input_files['file_global_param'])
    configWrite.set('input_files', 'file_cell_param', input_files['file_cell_param'])
    configWrite.set('input_files', 'file_rain', input_files['file_rain'])
    configWrite.set('input_files', 'file_et', input_files['file_et'])

    configWrite.add_section('output_files')
    configWrite.set('output_files', 'file_out', output_files['file_out'])
    configWrite.set('output_files', 'file_change_log_out', output_files['file_change_log_out'])
    configWrite.set('output_files', 'append_output', output_files['append_output'])

    configWrite.add_section('groups')
    configWrite.set('groups', 'group_name', 'sample_event')

    configWrite.add_section('external_flow')
    configWrite.set('external_flow', 'external_flow', external_flow['external_flow_status'])
    if external_flow['external_flow_status'].lower() == 'true':
        print ("External flow parameters need to be entered here")

    configWrite.add_section('numerical_options')
    configWrite.set('numerical_options', 'solve_s', numerical_options['solve_s'])
    configWrite.set('numerical_options', 'solve_o', numerical_options['solve_o'])
    configWrite.set('numerical_options', 'solve_c', numerical_options['solve_c'])
    configWrite.set('numerical_options', 'only_channel_output', numerical_options['only_channel_output'])

    configWrite.add_section('calib_params')
    configWrite.set('calib_params', 'fac_l', calib_param['fac_l'])
    configWrite.set('calib_params', 'fac_ks', calib_param['fac_ks'])
    configWrite.set('calib_params', 'fac_n_o', calib_param['fac_n_o'])
    configWrite.set('calib_params', 'fac_n_c', calib_param['fac_n_c'])
    configWrite.set('calib_params', 'fac_th_s', calib_param['fac_th_s'])

    with open(path_to_out_ini, 'wt') as configFile:
        configWrite.write(configFile)

    return path_to_out_ini


def get_variable_in_file_config_parser(section, variable, ini_file):
    from configparser import SafeConfigParser
    config = SafeConfigParser()
    config.read(ini_file)

    return config.get(section, variable)


def create_cell_param(create_file_ini_file, zero_slope_mngmt_ini_file):
    from pytopkapi.parameter_utils.create_file import generate_param_file
    from pytopkapi.parameter_utils import modify_file

    # Generate Parameter Files
    generate_param_file(create_file_ini_file, isolated_cells=False)
    print ("Cell Parameter file created")

    # slope corrections
    modify_file.zero_slope_management(zero_slope_mngmt_ini_file)
    print ("Zero Slope corrections made")

    return


def create_global_param(path_to_out_dat, A_thres=25000000, X=30.92208078, Dt=86400, W_min=1., W_Max=10.):
    title = ['X', 'Dt', 'Alpha_s', 'Alpha_o', 'Alpha_c', 'A_thres', 'W_min', 'W_max']
    values = [X, Dt, 2.5, 1.6666667, 1.6666667, A_thres, W_min, W_Max]
    values = [str(item) for item in values]
    with open(path_to_out_dat, "wb") as g_param_file:
        string = "\t\t".join(title) + '\n' + "\t\t".join(values)
        g_param_file.write(bytes(string, 'UTF-8'))
        # g_param_file.write(values)
    return path_to_out_dat


def create_rain_ET_file(total_no_of_cell, ppt_file_txt):
    # THIS IS TEST VERSION, it should not be used
    import h5py, numpy

    # output path
    rainfall_outputFile = "rainfields.h5"
    ET_outputFile = "ET.h5"

    # 1_del (for removing file readings)
    rain_from_file = numpy.genfromtxt(ppt_file_txt, delimiter='\t')

    time_step = rain_from_file.shape[0]
    no_of_cell = total_no_of_cell  # 3400
    # rainfall_intensity_perUnitTime = 20 #mm
    rainfall_reduction_factor = 1

    with h5py.File(rainfall_outputFile, 'w') as f2:

        f2.create_group('sample_event')
        f2['sample_event'].create_dataset('rainfall', shape=(time_step, no_of_cell), dtype='f')

        rainArray = f2['sample_event']['rainfall']
        data = numpy.zeros((time_step, no_of_cell))
        for i in range(time_step):
            # 2_del (for removing file readings)
            rainfall_intensity_perUnitTime = rain_from_file[i][-1] * rainfall_reduction_factor
            a = numpy.empty((1, no_of_cell))
            a.fill(rainfall_intensity_perUnitTime)  #
            data[i, :] = a

        # data[0,:] = numpy.zeros((1,no_of_cell))
        # data[time_step/4,:] = numpy.zeros((1,no_of_cell))
        # data[time_step/2,:] = numpy.zeros((1,no_of_cell))
        # data[time_step/3,:] = numpy.zeros((1,no_of_cell))
        rainArray[:] = data

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


def create_rain_ET_from_cell_no(simulation_folder, total_no_of_cell, ppt_file_txt=""):
    # THIS IS TEST VERSION, it should not be used
    import h5py, numpy, random

    # output path
    rainfall_outputFile = os.path.join(simulation_folder, "rainfields.h5")
    ET_outputFile = os.path.join(simulation_folder, "ET.h5")

    time_step = 30
    if ppt_file_txt != "":
        rain_from_file = numpy.genfromtxt(ppt_file_txt, delimiter='\t')
        time_step = rain_from_file.shape[0]

    no_of_cell = total_no_of_cell  # 3400
    # rainfall_intensity_perUnitTime = 20 #mm
    rainfall_reduction_factor = 1

    with h5py.File(rainfall_outputFile, 'w') as f2:

        f2.create_group('sample_event')
        f2['sample_event'].create_dataset('rainfall', shape=(time_step, no_of_cell), dtype='f')

        rainArray = f2['sample_event']['rainfall']
        data = numpy.zeros((time_step, no_of_cell))
        for i in range(time_step):
            rainfall_intensity_perUnitTime = random.uniform(0, 2)

            if ppt_file_txt != "":
                rainfall_intensity_perUnitTime = rain_from_file[i][-1] * rainfall_reduction_factor

            a = numpy.empty((1, no_of_cell))
            a.fill(rainfall_intensity_perUnitTime)  #
            data[i, :] = a

        # data[0,:] = numpy.zeros((1,no_of_cell))
        # data[time_step/4,:] = numpy.zeros((1,no_of_cell))
        # data[time_step/2,:] = numpy.zeros((1,no_of_cell))
        # data[time_step/3,:] = numpy.zeros((1,no_of_cell))
        rainArray[:] = data

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

    return rainfall_outputFile


def get_outletID_noOfCell(cell_param_file):
    cell_param_array = np.genfromtxt(cell_param_file, delimiter=' ')
    no_of_cell = cell_param_array.shape[0]

    # outlet_ID is the first element (cell lable) of parameter for the cell whose d/s id = -999
    outlet_ID = cell_param_array[cell_param_array[:, 14] < -99][0][0]
    return int(outlet_ID), no_of_cell


def zip_dir(path_dir, path_file_zip=''):
    if not path_file_zip:
        path_file_zip = os.path.join(
            os.path.dirname(path_dir), os.path.basename(path_dir) + '.zip')
    with zipfile.ZipFile(path_file_zip, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(path_dir):
            for file_or_dir in files + dirs:
                zip_file.write(
                    os.path.join(root, file_or_dir),
                    os.path.relpath(os.path.join(root, file_or_dir),
                                    os.path.join(path_dir, os.path.pardir)))


def change_date_from_mmddyyyy_to_yyyyddmm(in_date):
    '''
    :param in_date:         accepts date of formate '01/25/2010'
    :return:                converts the date to formate: '2010-01-25'
    '''
    from datetime import datetime
    in_date_element = datetime.strptime(in_date, '%m/%d/%Y')
    out_date = "%s-%s-%s" % (in_date_element.year, in_date_element.month, in_date_element.day)
    return out_date


def get_box_from_tif(input_raster, output_json):
    minx=  None
    miny = None
    maxx = None
    maxy = None

    try:
        temp_raster = 'temp.tif'
        os.system('gdalwarp -t_srs "+proj=longlat +ellps=WGS84" %s %s -overwrite' % (input_raster, temp_raster))

        ds = gdal.Open(temp_raster)
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + width * gt[4] + height * gt[5]
        maxx = gt[0] + width * gt[1] + height * gt[2]
        maxy = gt[3]

    except Exception as e:
        print (' Progres --> Error: Tiff files contents is not supported. Try another Tiff file', e)
        # return {'success': 'True', 'message':e}

    JSON_dict = {}
    JSON_dict['minx'] = minx
    JSON_dict['miny'] = miny
    JSON_dict['maxx'] = maxx
    JSON_dict['maxy'] = maxy

    # save
    with open(output_json, 'w') as newfile:
        json.dump(JSON_dict, newfile)

    return {'success':'True'}

def get_outlet_xy_from_shp(input_shp, output_json):
    # from shapely.geometry import shape
    import fiona

    temp_shp = 'temp.shp'
    os.system('ogr2ogr -t_srs EPSG:4326 %s  %s' % (temp_shp, input_shp))  # first output, second is input!

    # use fiona to get the bounds
    c = fiona.open(temp_shp)

    outlet_x = c.bounds[0]
    outlet_y = c.bounds[1]

    JSON_dict = {}
    JSON_dict['outlet_x'] = outlet_x
    JSON_dict['outlet_y'] = outlet_y

    # save
    with open(output_json, 'w') as newfile:
        json.dump(JSON_dict, newfile)

    return {'success':'True'}

def get_box_xyxy_from_shp(input_shp,output_json ):
    from shapely.geometry import shape
    import fiona

    temp_shp = 'temp.shp'
    os.system('ogr2ogr -t_srs EPSG:4326 %s  %s' % (temp_shp, input_shp))  # first output, second is input!

    c = fiona.open(temp_shp)

    # first record
    first_shape = c.next()

    # shape(first_shape['geometry']) -> shapely geometry

    maxy = shape(first_shape['geometry']).bounds[3]
    miny = shape(first_shape['geometry']).bounds[1]
    maxx = shape(first_shape['geometry']).bounds[2]
    minx = shape(first_shape['geometry']).bounds[0]

    JSON_dict = {}
    JSON_dict['minx'] = minx
    JSON_dict['miny'] = miny
    JSON_dict['maxx'] = maxx
    JSON_dict['maxy'] = maxy

    # save
    with open(output_json, 'w') as newfile:
        json.dump(JSON_dict, newfile)

    return {'success': 'True'}


def change_timestep_of_forcing_netcdf(input_netcdf,  output_rain_fname, output_et_reference_fname, time_interval_in_hr=6):
    from netCDF4 import Dataset
    root = Dataset(input_netcdf)
    rain_ar = root.variables['SWIT'][:]
    root.close()

    print ('Progress >> Input netCDF read. The dimension is ', rain_ar.shape)

    timestep_factor =  int(24/int(time_interval_in_hr)) # time interval should divide 24 without leaving any remainders
    new_rain_ar = np.zeros(  (  int(rain_ar.shape[0])/timestep_factor ,  int(rain_ar.shape[1]) , int(rain_ar.shape[2])  )  )

    for i in range(0,len(rain_ar),timestep_factor):
        k = 1
        sum = np.zeros((   int(rain_ar.shape[1]) , int(rain_ar.shape[2])   ))
        for j in range(timestep_factor):
            sum = sum + rain_ar[i+j]
            k = k +1
            new_rain_ar[i/timestep_factor] = sum/k

    print ('Progress >> "Converted" Rain successfully created. The dimension is ', new_rain_ar.shape)

    shutil.copy(input_netcdf, output_rain_fname)
    shutil.copy(input_netcdf, output_et_reference_fname)


    # # # FOR RAIN # # #
    root = Dataset(output_rain_fname, 'a')
    root.variables['SWIT'] = np.zeros(  (  int(new_rain_ar.shape[0]) ,  int(new_rain_ar.shape[1]) , int(new_rain_ar.shape[2])  )  )
    root.variables['SWIT'][:] = new_rain_ar
    print (' NEW ppt dimension', root.variables['SWIT'][:].shape)
    root.close()


    # instead of creating a fresh netCDF for ET file, we use Tmax file, and insert caluclated reference ET values on it
    cmd = "ncrename -v SWIT,prcp " + output_rain_fname
    os.system(cmd)

    # change units for the renamed variable to mm/day
    cmd = "ncatted -a units,prcp,m,c,'mm/day' " + output_rain_fname
    os.system(cmd)

    # change long_name for the renamed variable to 'short crop reference evapotranspiration'
    cmd = "ncatted -a long_name,prcp,m,c,'snowmelt and rain summed' " + output_rain_fname
    os.system(cmd)

    # change long_name for the renamed variable to 'short crop reference evapotranspiration'
    cmd = "ncatted -a cell_methods,prcp,m,c,'mean daily' " + output_rain_fname
    os.system(cmd)



    # # # FOR ET # # #
    root = Dataset(output_et_reference_fname, 'a')
    root.variables['SWIT'] = np.zeros((int(rain_ar.shape[0]) / timestep_factor, int(rain_ar.shape[1]), int(rain_ar.shape[2])))  #create empyt numpy array as a variable
    root.variables['SWIT'][:] = new_rain_ar*0.0
    root.close()

    # instead of creating a fresh netCDF for ET file, we use Tmax file, and insert caluclated reference ET values on it
    cmd = "ncrename -v SWIT,ETr " + output_et_reference_fname
    os.system(cmd)

    # change units for the renamed variable to mm/day
    cmd = "ncatted -a units,ETr,m,c,'mm/day' " + output_et_reference_fname
    os.system(cmd)

    # change long_name for the renamed variable to 'short crop reference evapotranspiration'
    cmd = "ncatted -a long_name,ETr,m,c,'short crop reference evapotranspiration' " + output_et_reference_fname
    os.system(cmd)

    # change long_name for the renamed variable to 'short crop reference evapotranspiration'
    cmd = "ncatted -a cell_methods,ETr,m,c,'mean daily' " + output_et_reference_fname
    os.system(cmd)


    return {'success': 'True'}



def download_pcp(startDate, endDate, cell_size, input_watershed, climate_variable='prcp'):
    # :TODO Currently=>USES hydrogate_python_client to get the nc file from production level HydroDS
    # Change it later on, when this is published to production HydroDS, dont need this
    from datetime import datetime
    from hydrogate import HydroDS
    import settings
    HDS = HydroDS(username=settings.USER_NAME, password=settings.PASSWORD)

    startYear = datetime.strptime(startDate, "%m/%d/%Y").year
    endYear = datetime.strptime(endDate, "%m/%d/%Y").year

    Watershed_temp = HDS.raster_to_netcdf(input_watershed, output_netcdf='watershed' + str(cell_size) + '.nc')

    # In the netCDF file rename the generic variable "Band1" to "watershed"
    Watershed_NC = HDS.netcdf_rename_variable(input_netcdf_url_path=Watershed_temp['output_netcdf'],
                                              output_netcdf='watershed.nc', input_variable_name='Band1',
                                              output_variable_name='watershed')
    climate_Vars = [climate_variable]
    #### iterate through climate variables
    for var in climate_Vars:
        for year in range(startYear, endYear + 1):
            climatestaticFile1 = var + "_" + str(year) + ".nc4"
            climateFile1 = var + "_" + str(year) + ".nc"
            Year1sub_request = HDS.subset_netcdf(input_netcdf=climatestaticFile1,
                                                 ref_raster_url_path=input_watershed,
                                                 output_netcdf=climateFile1)
            concatFile = "conc_" + climateFile1
            if year == startYear:
                concatFile1_url = Year1sub_request['output_netcdf']
            else:
                concatFile2_url = Year1sub_request['output_netcdf']
                concateNC_request = HDS.concatenate_netcdf(input_netcdf1_url_path=concatFile1_url,
                                                           input_netcdf2_url_path=concatFile2_url,
                                                           output_netcdf=concatFile)
                concatFile1_url = concateNC_request['output_netcdf']

        timesubFile = "tSub_" + climateFile1
        subset_NC_by_time_result = HDS.subset_netcdf_by_time(input_netcdf_url_path=concatFile1_url,
                                                             time_dimension_name='time', start_date=startDate,
                                                             end_date=endDate, output_netcdf=timesubFile)
        subset_NC_by_time_file_url = subset_NC_by_time_result['output_netcdf']
        if var == 'prcp':
            proj_resample_file = var + "_0.nc"
        else:
            proj_resample_file = var + "0.nc"
        ncProj_resample_result = HDS.project_subset_resample_netcdf(input_netcdf_url_path=subset_NC_by_time_file_url,
                                                                    ref_netcdf_url_path=Watershed_NC['output_netcdf'],
                                                                    variable_name=var, output_netcdf=proj_resample_file)
        ncProj_resample_file_url = ncProj_resample_result['output_netcdf']

        #### Do unit conversion for precipitation (mm/day --> m/hr)
        if var == 'prcp':
            proj_resample_file = var + "0.nc"
            ncProj_resample_result = HDS.convert_netcdf_units(input_netcdf_url_path=ncProj_resample_file_url,
                                                              output_netcdf=proj_resample_file,
                                                              variable_name=var, variable_new_units='m/hr',
                                                              multiplier_factor=0.00004167,
                                                              offset=0.0)  #:TODO change the units to mm or m /day
            ncProj_resample_file_url = ncProj_resample_result['output_netcdf']

    return ncProj_resample_file_url


def abstract_pcp(startDate, endDate, input_watershed, cell_size=None, climate_variable='prcp',
                 output_nc_fname='ppt.nc'):
    """

    :param startDate:       format accepted mm/dd/yy
    :param endDate:         format accepted mm/dd/yy
    :param input_watershed: tif
    :param cell_size:       integer ( float might work too)
    :param climate_variable:string. Accepted values are 'tmin', 'tmax', 'srad', 'prcp'. Default is 'prcp'
    :param output_nc_fname:
    :return:
    """
    # does not use any other HydroDS, usese the data stored in the running HydroDS
    from datetime import datetime

    startYear = datetime.strptime(startDate, "%m/%d/%Y").year
    endYear = datetime.strptime(endDate, "%m/%d/%Y").year

    start_time_index = datetime.strptime(startDate, "%m/%d/%Y").day
    end_time_index = datetime.strptime(startDate, "%m/%d/%Y").day + (
    datetime.strptime(endDate, "%m/%d/%Y") - datetime.strptime(startDate, "%m/%d/%Y")).days

    if cell_size is None:
        cell_size = get_cellSize(input_watershed)

    rasterToNetCDF_rename_variable(input_watershed, output_netcdf='watershed' + str(cell_size) + '.nc')

    # In the netCDF file rename the generic variable "Band1" to "watershed"
    Watershed_NC_out = 'watershed.nc'
    Watershed_NC = netCDF_rename_variable(input_netcdf='watershed' + str(cell_size) + '.nc',
                                          output_netcdf=Watershed_NC_out, input_varname='Band1',
                                          output_varname='watershed')
    climate_Vars = [climate_variable]
    #### iterate through climate variables
    for var in climate_Vars:
        for year in range(startYear, endYear + 1):
            climatestaticFile1 = '/home/ahmet/hydosdata/DaymetClimate/' + var + "_" + str(year) + ".nc4"
            climateFile1 = var + "_" + str(year) + ".nc"
            Year1sub_request = subset_netCDF_to_reference_raster(input_netcdf=climatestaticFile1,
                                                                 reference_raster=input_watershed,
                                                                 output_netcdf=climateFile1)
            concatFile = "conc_" + climateFile1
            if year == startYear:
                concatFile1_url = climateFile1
            else:
                concatFile2_url = climateFile1
                concateNC_request = concatenate_netCDF(input_netcdf1=concatFile1_url,
                                                       input_netcdf2=concatFile2_url,
                                                       output_netcdf=concatFile)
                concatFile1_url = concatFile

        timesubFile = "tSub_" + climateFile1
        subset_NC_by_time_result = get_netCDF_subset_TimeDim(input_netcdf=concatFile1_url,
                                                             time_dim_name='time', start_time_index=start_time_index,
                                                             end_time_index=end_time_index, output_netcdf=timesubFile)
        subset_NC_by_time_file_url = timesubFile
        if var == 'prcp':
            proj_resample_file = var + "_0.nc"  # output_nc_fname
        else:
            proj_resample_file = output_nc_fname  # var + "0.nc"
        ncProj_resample_result = project_subset_and_resample_netcdf_to_reference_netcdf(
            input_netcdf=subset_NC_by_time_file_url,
            reference_netcdf=Watershed_NC_out,
            variable_name=var, output_netcdf=proj_resample_file)
        ncProj_resample_file_url = proj_resample_file

        #### Do unit conversion for precipitation (mm/day --> m/hr)
        if var == 'prcp':
            proj_resample_file = output_nc_fname  # var + "0.nc"
            ncProj_resample_result = convert_netcdf_units(input_netcdf=ncProj_resample_file_url,
                                                          output_netcdf=proj_resample_file,
                                                          variable_name=var, variable_new_units='m/hr',
                                                          multiplier_factor=0.00004167,
                                                          offset=0.0)  #:TODO change the units to mm or m /day
            ncProj_resample_file_url = proj_resample_file

    # delete all the other temporary nc files
    onlyfiles = [f for f in os.listdir() if os.path.isfile(f)]
    unnecessary_nc_fullpath = [f for f in onlyfiles if f != output_nc_fname]
    for file in unnecessary_nc_fullpath:
        os.remove(file)

    return output_nc_fname


def abstract_climate(startDate, endDate, input_raster, cell_size=None,
                     output_vp_fname='output_vp.nc', output_tmin_fname='output_tmin.nc',
                     output_tmax_fname='output_tmax.nc',
                     output_srad_fname='output_srad.nc',
                     output_prcp_fname='output_prcp.nc'):  # , output_dayl_fname='output_dayl.nc'
    """

    :param startDate:       format accepted mm/dd/yyyy
    :param endDate:         format accepted mm/dd/yyyy
    :param input_watershed: tif
    :param cell_size:       integer ( float might work too)
    :param climate_variable:string. Accepted values are 'tmin', 'tmax', 'srad', 'prcp'. 'ALL' is accepted if all the files are desired . Default is 'prcp'
    :param output_nc_fname:
    :return:
    """
    # does not use any other HydroDS, usese the data stored in the running HydroDS
    print ('Step0')
    from datetime import datetime

    input_watershed = input_raster

    startYear = datetime.strptime(startDate, "%m/%d/%Y").year
    endYear = datetime.strptime(endDate, "%m/%d/%Y").year

    start_time_index = datetime.strptime(startDate, "%m/%d/%Y").day
    end_time_index = datetime.strptime(startDate, "%m/%d/%Y").day + (
    datetime.strptime(endDate, "%m/%d/%Y") - datetime.strptime(startDate, "%m/%d/%Y")).days

    print ('Step1')
    if cell_size is None:
        cell_size = int(get_cellSize(input_watershed))

    rasterToNetCDF_rename_variable(input_watershed, output_netcdf='watershed0.nc')
    print ('Step2')

    # In the netCDF file rename the generic variable "Band1" to "watershed"
    Watershed_NC_out = 'watershed.nc'
    Watershed_NC = netCDF_rename_variable(input_netcdf='watershed0.nc',
                                          output_netcdf=Watershed_NC_out, input_varname='Band1',
                                          output_varname='watershed')
    print ('Step3')
    climate_Vars = ['vp', 'tmin', 'tmax', 'srad', 'prcp']  # , 'dayl'
    for var in climate_Vars:
        for year in range(startYear, endYear + 1):
            climatestaticFile1 = '/home/ahmet/hydosdata/DaymetClimate/' + var + "_" + str(year) + ".nc4"
            climateFile1 = var + "_" + str(year) + ".nc"
            Year1sub_request = subset_netCDF_to_reference_raster(input_netcdf=climatestaticFile1,
                                                                 reference_raster=input_watershed,
                                                                 output_netcdf=climateFile1)
            concatFile = "conc_" + climateFile1
            if year == startYear:
                concatFile1_url = climateFile1
            else:
                concatFile2_url = climateFile1
                concateNC_request = concatenate_netCDF(input_netcdf1=concatFile1_url,
                                                       input_netcdf2=concatFile2_url,
                                                       output_netcdf=concatFile)
                concatFile1_url = concatFile

        timesubFile = "tSub_" + climateFile1
        subset_NC_by_time_result = get_netCDF_subset_TimeDim(input_netcdf=concatFile1_url,
                                                             time_dim_name='time', start_time_index=start_time_index,
                                                             end_time_index=end_time_index, output_netcdf=timesubFile)
        subset_NC_by_time_file_url = timesubFile

        # name the output file
        if var == 'prcp':
            proj_resample_file = output_prcp_fname
        elif var == 'tmin':
            proj_resample_file = output_tmin_fname
        elif var == 'tmax':
            proj_resample_file = output_tmax_fname
        elif var == 'srad':
            proj_resample_file = output_srad_fname
        elif var == 'vp':
            proj_resample_file = output_vp_fname
        # elif var == 'dayl':
        #     proj_resample_file = output_dayl_fname



        ncProj_resample_result = project_subset_and_resample_netcdf_to_reference_netcdf(
            input_netcdf=subset_NC_by_time_file_url,
            reference_netcdf=Watershed_NC_out,
            variable_name=var, output_netcdf=proj_resample_file)
        ncProj_resample_file_url = proj_resample_file

    print ('Step4')
    required_files_list = [output_prcp_fname, output_tmin_fname, output_tmax_fname, output_srad_fname, output_vp_fname]

    # delete all the other temporary nc files
    onlyfiles = [f for f in os.listdir() if os.path.isfile(f)]
    # unnecessary_nc_fullpath = [f for f in onlyfiles if f != output_nc_fname]

    # for file in  onlyfiles: #unnecessary_nc_fullpath:
    #     if file not in required_files_list:
    #         os.remove(file)

    return {'success': 'True'}


def calculate_reference_et_from_netCDFs(dem_nc, srad_nc, tmax_nc, tmin_nc, out_et_nc=None, vp_nc=None, ws=2):
    """
    reference evapotranspiration (ETo) from a hypothetical short grass (0.12m)reference surface, and surface resistance
    of 70s/m, and an albedo of 0.23 using the FAO-56 Penman-Monteith equation, from pyeto python package

    The reference surface resembles an extensive surface of green, well-watered grass of uniform height,
    actively growing and completely shading the ground, and
    moderately dry soil surface resulting from about a weekly irrigation frequency.
    :param dem_nc:          NetCDF file for dem, with values in m above sea level in variable='elevation' (2D array)
    :param srad_nc:         NetCDF file for shortwave radiation in degree Watt/m2   (3D array)
    :param tmax_nc:         NetCDF file for of maximum temperature in degree celius (3D array)
    :param tmin_nc:         NetCDF file for of minimum temperature in degree celius (3D array)
    :param vp_nc:           NOT USED!
    :param ws:              Wind speed in m/s    default 2m/s
    :return:                3D array (and netcdf file) of reference ET in mm/day
    """
    from pyeto import fao
    import pyeto
    from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
    import numpy as np

    # vectorized function to work with array
    svp_vec = np.vectorize(fao.svp_from_t)  # 3d
    avp_vec = np.vectorize(fao.avp_from_tmin)  # 3d
    psy_vec = np.vectorize(fao.psy_const)  # 2d # will it work
    atm_pressure_vec = np.vectorize(fao.atm_pressure)
    delta_svp_vec = np.vectorize(fao.delta_svp)  # 3d
    fao56_penman_monteith_vec = np.vectorize(fao.fao56_penman_monteith)
    # sol_dec_vec = np.vectorize(fao.sol_dec)
    # sunset_hour_angle_vec = np.vectorize(fao.sunset_hour_angle)
    # inv_rel_dist_earth_sun_vec = np.vectorize(fao.inv_rel_dist_earth_sun)
    # et_rad_vec = np.vectorize(fao.et_rad)
    # cs_rad_vec = np.vectorize(fao.cs_rad)
    # avp_from_tmin_vec = np.vectorize(fao.avp_from_tmin)
    # net_out_lw_rad_vec = np.vectorize(fao.net_out_lw_rad)
    # net_rad_vec = np.vectorize(fao.net_rad)

    # dem
    root = Dataset(dem_nc, 'r')  # Dataset is the class behavior to open the file
    ar_dem = root.variables['elevation'][:]
    root.close()

    # tmax
    root = Dataset(tmax_nc, 'r')
    ar_tmax = root.variables['tmax'][:]
    avg_latitude = root.variables['lat'][:].mean() * 3.14 / 180.0
    ar_time_num_after_1980 = root.variables['time'][:]
    root.close()

    # tmin
    root = Dataset(tmin_nc, 'r')
    ar_tmin = root.variables['tmin'][:]
    root.close()

    print ("Progress >> ET calculations: Files read successfully")

    # different function of fao requires temperature in different unit . Yes, it sucks!
    t_celsius = (ar_tmin + ar_tmax) / 2.0  # 3d
    t_kelvin = 273.15 + (ar_tmin + ar_tmax) / 2.0  # 3d

    # vp: Actual vapor pressure
    if vp_nc != None:
        root = Dataset(vp_nc, 'r')
        ar_avp = root.variables['vp'][:] / 1000.0  # converting Pa to Kpa
        root.close()
    else:
        ar_avp = avp_vec(ar_tmin)

    # Some of the functions use vectorized equations, so they cannot be above the vectorized formula
    def total_daily_sol_rad_from_swrad(srad_nc, latitude=None):
        root = Dataset(srad_nc, 'r')
        ar_srad = root.variables['srad'][:]
        ar_time = root.variables['time'][:]
        if latitude is None:
            latitude = root.variables['lat'][:].mean() * 3.14 / 180.0
        root.close()

        # current_timestep_date = datetime.datetime(1980, 1, 1) + timedelta(days=109652)  # https://media.readthedocs.org/pdf/pyeto/latest/pyeto.pdf
        # current_day_of_yr = current_timestep_date.timetuple().tm_yday

        # Iterate for timestep. For each time step, identify which day of the year it is
        ar_current_timestep_date = np.array([(datetime(1980, 1, 1) + timedelta(days=int(t))) for t in ar_time])
        ar_current_day_of_yr = np.array([one_date.timetuple().tm_yday for one_date in ar_current_timestep_date])
        hr_in_day = np.array(
            [fao.daylight_hours(fao.sunset_hour_angle(latitude=latitude, sol_dec=fao.sol_dec(day_of_year=t))) for t in
             ar_current_day_of_yr])

        # for each timestep, convert the srad from W/m2 to MJ / m2 / day
        for i in range(len(ar_srad)):
            ar_srad[i] = ar_srad[i] * hr_in_day[
                i] * 3600.0 / 1000000.0  # https://daac.ornl.gov/DAYMET/guides/Daymet_mosaics.html

        total_daily_sol_rad = ar_srad  # the value has now been converted from 	W/m2  to   MJ /m2 /day
        return total_daily_sol_rad

    def net_radiation(ar_total_incoming_sol_rad, ar_current_day_of_yr, ar_tmin, ar_tmax, ar_altitude, latitude,
                      ar_avp=None):
        """
        :param total_incoming_sol_rad: 3D array of net incoming solar radiation (MJ /m2 /d). From daymet swrad nc file
        :param tmin:            3D array of temperature in celsius. Obtained from Daymet netcdf files
        :param tmax:            "               "           "
        :param altitude:        , in radians
        :param latitude:        2D array of election above sea level in meters
        :param ar_current_day_of_yr:         1D array, with length = timesteps, of value between 1-365 (not necessarily integer)
        :param ar_avp:          3D array, actual vapor pressure in KPa.Remember, daymet has in Pa. Change before this fn
        :return:                3D array of net radiation in (MJ /m2 /d) according to daymet swrad
        """

        sol_dec = fao.sol_dec(ar_current_day_of_yr)
        sha = fao.sunset_hour_angle(latitude, sol_dec)  # sol_dec in radians
        ird = fao.inv_rel_dist_earth_sun(ar_current_day_of_yr)  # returns dimensionless parameter
        et_rad = fao.et_rad(latitude, sol_dec, sha, ird)  # extraterrestrial radiation. sol_dec, sha in radians
        cs_rad = fao.cs_rad(ar_altitude, et_rad)  # cs: clear sky radiation

        if ar_avp is None:
            ar_avp = fao.avp_from_tmin(ar_tmin)  # in deg C

        no_lw_rad = fao.net_out_lw_rad(ar_tmin + 273.15, ar_tmax + 273.15, ar_total_incoming_sol_rad, cs_rad,
                                       ar_avp)  # in deg K. Actual vap pressure avp in kpa required.
        net_radiation = fao.net_rad(ni_sw_rad=ar_total_incoming_sol_rad, no_lw_rad=no_lw_rad)  # in MJ /m2 /day
        values = {'sol_dec': sol_dec, 'sha': sha, 'ird': ird, 'et_rad': et_rad, 'cs_rad': cs_rad, 'ar_avp': ar_avp}
        if net_radiation > 50:
            print (values)

        return net_radiation

    def net_radiation_backup(ar_total_incoming_sol_rad, ar_current_day_of_yr, ar_tmin, ar_tmax, ar_altitude, latitude,
                             ar_avp=None):
        """
        :param total_incoming_sol_rad: 3D array of net incoming solar radiation (MJ /m2 /d). From daymet swrad nc file
        :param tmin:            3D array of temperature in celsius. Obtained from Daymet netcdf files
        :param tmax:            "               "           "
        :param altitude:        , in radians
        :param latitude:        2D array of election above sea level in meters
        :param ar_current_day_of_yr:         1D array, with length = timesteps, of value between 1-365 (not necessarily integer)
        :param ar_avp:          3D array, actual vapor pressure in KPa.Remember, daymet has in Pa. Change before this fn
        :return:                3D array of net radiation in (MJ /m2 /d) according to daymet swrad
        """

        sol_dec = sol_dec_vec(ar_current_day_of_yr);
        print ("Net Rad Calc >> sol_dec (solar declination) ")
        sha = sunset_hour_angle_vec(latitude, sol_dec);
        print ("Net Rad Calc >> sha (sun hr angle)")  # sol_dec in radians
        ird = inv_rel_dist_earth_sun_vec(ar_current_day_of_yr);
        print ("Net Rad Calc >> ird (inverse dist earth sun)")  # returns dimensionless parameter
        et_rad = et_rad_vec(latitude, sol_dec, sha, ird);
        print (
        "Net Rad Calc >> et_rad (extraterrestial rad)")  # et_rad=extraterrestrial radiation. sol_dec, sha in radians
        cs_rad = cs_rad_vec(ar_altitude, et_rad);
        print ("Net Rad Calc >> cs_rad (clear sky radiation)")  # cs: clear sky radiation

        if ar_avp is None:
            ar_avp = avp_from_tmin_vec(ar_tmin)  # in deg C

        no_lw_rad = net_out_lw_rad_vec(ar_tmin + 273.15, ar_tmax + 273.15, ar_total_incoming_sol_rad, cs_rad,
                                       ar_avp)  # in deg K. Actual vap pressure avp in kpa required.
        print ("Net Rad Calc >> no_lw_rad (Long wave radiation)")
        net_rad = net_rad_vec(ar_total_incoming_sol_rad, no_lw_rad)  # in MJ /m2 /day
        print ("Net Rad Calc >> net_rad (net radiation)")
        return net_rad

    # vectorize the function just created
    net_radiation_vec = np.vectorize(net_radiation)

    # # # # # # # # ACTUAL CALCULATIONS START HERE # # # # # # # #
    # total daily short wave radiation [Mj / m2/ day]
    ar_total_incoming_sol_rad = total_daily_sol_rad_from_swrad(srad_nc=srad_nc, latitude=avg_latitude)  # 3d
    print ("Progress >> srad array converted from Watt/m2 to MJ / m2/ day")

    # timestep array based on NetCDF file
    ar_current_timestep_date = np.array(
        [(datetime(1980, 1, 1) + timedelta(days=t)) for t in ar_time_num_after_1980.tolist()])

    # convert the timestep to 1d array
    ar_current_day_of_yr = np.array([one_date.timetuple().tm_yday for one_date in ar_current_timestep_date])

    # converting array of day of year from 1D, to 3D. i.e. from one value for one timestep, to one value for one cell
    ar_3d_current_day_of_yr = np.ones(ar_tmax.shape)
    j = 0
    for i in ar_current_day_of_yr:
        ar_3d_current_day_of_yr[j] = i * ar_3d_current_day_of_yr[j]  # (ar_tmax[0].shape)
        j = j + 1

    net_radiation_calc = net_radiation_vec(ar_total_incoming_sol_rad=ar_total_incoming_sol_rad,
                                           ar_current_day_of_yr=ar_3d_current_day_of_yr,  # ar_current_day_of_yr,
                                           ar_tmin=ar_tmin, ar_tmax=ar_tmax, ar_altitude=ar_dem, latitude=avg_latitude,
                                           ar_avp=ar_avp)

    print ("Progress >> Net Radiation calculated successfully: ")

    eto = fao56_penman_monteith_vec(net_rad=net_radiation_calc, t=t_kelvin, ws=ws, svp=svp_vec(t_celsius), avp=ar_avp,
                                    delta_svp=delta_svp_vec(t_celsius),
                                    psy=psy_vec(atm_pressure_vec(ar_dem)))  # ,t,ws,svp,avp,delta_svp,psy )

    # eto = eto*1000.0 # converting to meters/day from mm/day
    # saving file as netCDF file
    try:
        # :TODO copy tmin netcdf file, and rename in ET.nc
        import shutil
        shutil.copy(tmax_nc, out_et_nc)

        root = Dataset(out_et_nc, 'a')

        # replace the tmin array with ET array created
        root.variables['tmax'][:] = eto
        root.close()
        print ('Progress >> Reference ET file successfully created')
    except Exception as e:
        print ('Progress >> Reference ET 3d array created, but could not write ET to file' + str(e))
        return eto

    print ('Reference ET was calculated. The values are- max: %s, min: %s and mean: %s values are' % (
    eto.max(), eto.min(), eto.mean()))
    return eto


def calculate_rain_ET_from_daymet(startDate, endDate, input_raster, input_dem, cell_size=None,
                                  output_et_reference_fname='ET_reference.nc', output_rain_fname='rain.nc'):
    """

    :param startDate:
    :param endDate:
    :param input_raster:
    :param input_dem:
    :param cell_size:
    :param output_et_reference_fname:
    :param output_rain_fname:           :TODO units
    :return:
    """
    dir = os.path.split(output_et_reference_fname)[0]
    os.chdir(dir)
    print ("Progress >> ET calculations will begin shortly. Final netcdf file will be saved in" + dir)

    abstract_climate(startDate=startDate, endDate=endDate, input_raster=input_raster, cell_size=cell_size,
                     output_vp_fname=dir + '/output_vp.nc', output_tmin_fname=dir + '/output_tmin.nc',
                     output_tmax_fname=dir + '/output_tmax.nc', output_srad_fname=dir + '/output_srad.nc',
                     output_prcp_fname=output_rain_fname)

    rasterToNetCDF_rename_variable(input_dem, output_netcdf=dir + '/watershed0.nc')

    netCDF_rename_variable(input_netcdf=dir + '/watershed0.nc', output_netcdf=dir + '/watershed.nc',
                           input_varname='Band1', output_varname='elevation')

    calculate_reference_et_from_netCDFs(dem_nc=dir + '/watershed.nc', srad_nc=dir + '/output_srad.nc',
                                        tmax_nc=dir + '/output_tmax.nc', tmin_nc=dir + '/output_tmin.nc',
                                        vp_nc=dir + '/output_vp.nc', out_et_nc=output_et_reference_fname)

    print ("Progress >> ET calculations successful. Now preparing the final netCDF outputs")

    # instead of creating a fresh netCDF for ET file, we use Tmax file, and insert caluclated reference ET values on it
    cmd = "ncrename -v tmax,ETr " + output_et_reference_fname
    os.system(cmd)

    # change units for the renamed variable to mm/day
    cmd = "ncatted -a units,ETr,m,c,'mm/day' " + output_et_reference_fname
    os.system(cmd)

    # change long_name for the renamed variable to 'short crop reference evapotranspiration'
    cmd = "ncatted -a long_name,ETr,m,c,'short crop reference evapotranspiration' " + output_et_reference_fname
    os.system(cmd)

    # change long_name for the renamed variable to 'short crop reference evapotranspiration'
    cmd = "ncatted -a cell_methods,ETr,m,c,'mean daily' " + output_et_reference_fname
    os.system(cmd)

    return {'success': 'True'}


def create_pytopkapi_hdf5_from_nc(nc_f, mask_tiff, output_folder=""):
    import h5py, numpy
    from netCDF4 import Dataset

    root = Dataset(nc_f, 'r')
    ppt = root.variables['prcp'][:]  # all the precipitation records, in 3d array (time * x * y)

    dset = gdal.Open(mask_tiff)
    mask = dset.ReadAsArray()  # a (x*y) array of values from raster

    time_step = ppt.shape[0]  # time length of the rainfall 3d array
    no_of_cell = mask[mask == 1].size  # mask[mask==1] creates a 1d array satisfying condition mask==1

    # output path
    rainfall_outputFile = os.path.join(output_folder, "rainfields.h5")
    ET_outputFile = os.path.join(output_folder, "ET.h5")

    # import shutil
    # shutil.copy2('/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/rainfields.h5', output_folder)
    # shutil.copy2('/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/ET.h5', output_folder)

    with h5py.File(rainfall_outputFile, 'w') as f2:
        # f2.create_group(u'sample_event')
        # f2[u'sample_event'].create_dataset(u'rainfall', shape=(time_step, no_of_cell), dtype='f')
        # grp0 = f2.create_group('test')
        print ('H5py description', str(h5py))
        print ('H5py create_group description', str(f2.create_group))

        daataa = numpy.zeros((time_step, no_of_cell))
        for i in range(time_step):
            ppt_at_that_time_step = ppt[i]
            daataa[i, :] = ppt_at_that_time_step[mask == 1]

        group_name = 'sample_event/rainfall'.encode('utf-8')
        dset = f2.create_dataset(group_name, data=daataa)
        # dset = f2.create_dataset("sample_event/rainfall", data= daataa, dtype="S10")

    with h5py.File(rainfall_outputFile, 'w') as f2:
        # f2.create_group(u'sample_event')
        # f2[u'sample_event'].create_dataset(u'rainfall', shape=(time_step, no_of_cell), dtype='f')
        # grp0 = f2.create_group('test')
        print ('H5py description', str(h5py))
        print ('H5py create_group description', str(f2.create_group))

        # grp = f2.create_group('sample_event')  #.encode('utf-8')
        group_name = 'sample_event/rainfall'.encode('utf-8')
        f2.create_dataset(group_name, shape=(time_step, no_of_cell), dtype='f')

        rainArray = f2[u'sample_event'][u'rainfall']

        data = numpy.zeros((time_step, no_of_cell))
        for i in range(time_step):
            ppt_at_that_time_step = ppt[i]
            data[i, :] = ppt_at_that_time_step[mask == 1]

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

    return rainfall_outputFile


def change_ini_path_to_local2(folders_with_inis):
    onlyfiles = [f for f in os.listdir(folders_with_inis) if os.path.isfile(os.path.join(folders_with_inis, f))]
    inis_fullpath = [os.path.join(folders_with_inis, f) for f in onlyfiles if f.endswith('.ini')]
    for ini_file in inis_fullpath:
        config = SafeConfigParser()
        config.read(ini_file)

        # somehow replace /.../ to /


def change_ini_path_to_local(folders_with_inis, string_to_change_in_files=''):
    def inplace_change(filename, old_string, new_string):
        # Safely read the input filename using 'with'
        with open(filename) as f:
            s = f.read()
            if old_string not in s:
                # print '"{old_string}" not found in {filename}.'.format(**locals())
                return

        # Safely write the changed content, if found in the file
        with open(filename, 'w') as f:
            # print 'Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals())
            s = s.replace(old_string, new_string)
            f.write(s)

    onlyfiles = [f for f in os.listdir(folders_with_inis) if os.path.isfile(os.path.join(folders_with_inis, f))]
    inis_fullpath = [os.path.join(folders_with_inis, f) for f in onlyfiles if f.endswith('.ini')]
    for ini_file in inis_fullpath:
        inplace_change(ini_file, string_to_change_in_files, '')


def return_data_line(afile= 'Q_raw.txt'):
    f = open(afile, 'r')
    i = 1
    for line in f.readlines():
      if line[0:4]=='USGS':
        print (i)
        return i-1
      i = i +1
    return i-1

def downloadandresampleusgsdischarge(USGS_Gage, begin_date='10/01/2010', end_date='12/30/2010',
                                     out_fname='q_obs_cfs.txt',
                                     output_unit='cfs', resampling_time='1D', resampling_method='mean'):
    """
    Downloads, and then resamples the discharge data from USGS using the url of the format:
    http://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&format=rdb&site_no=10109000&period=&begin_date=2015-10-01&end_date=2015-10-31
    INPUT:
    USGS_Gage :     string, e.g. 10109000
    begin_date=     string, e.g. '10/01/2010'
    end_date=       string, e.g. ''12/30/2010'
    out_fname=      string, e.g. 'Q_cfs.txt'
    output_unit=    string, e.g. 'cfs' or 'cumecs'
    resampling_time=  string, e.g. '1D'
    resampling_method=string, e.g.'mean'
    """


    begin_date = change_date_from_mmddyyyy_to_yyyyddmm(in_date=begin_date)
    end_date = change_date_from_mmddyyyy_to_yyyyddmm(in_date=end_date)
    urlString3 = 'http://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&format=rdb&site_no=%s&period=&begin_date=%s&end_date=%s' % (
    USGS_Gage, begin_date, end_date)

    response = urllib.request.urlopen(urlString3)  # instance of the file from the URL
    html = response.read()  # reads the texts into the variable html

    print ('Progress --> data read from the url ', urlString3)
    with open('Q_raw.txt', 'wb') as f:
        f.write(html)

    rows_to_skip = return_data_line(afile='Q_raw.txt')
    df = pd.read_csv('Q_raw.txt', delimiter='\t', skiprows=rows_to_skip,
                     names=['agency_cd', 'USGS_Station_no', 'datatime', 'timezone', 'Q_cfs', 'Quality'])

    # convert datetime from string to datetime
    df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2], errors='ignore')

    # create a different dataframe with just the values and datetime
    df_datetime_val = df[['datatime', 'Q_cfs']]

    # convert the values to series
    values = []
    dates = []

    # add values to the list a
    multiplier = 1.0
    for v in df_datetime_val.iloc[:, 1]:

        if output_unit.lower() == 'cumecs' or output_unit.lower() == 'cumec':
            multiplier = 0.028316846592

        values.append (round(float(v) * multiplier, 6) )

    # add datatime to list b
    for v in df_datetime_val.iloc[:, 0]:
        dates.append(v)

    # prepare a panda series
    ts = pd.Series(values, index=dates)

    # resample to daily or whatever
    # ts_mean = ts.resample('1D', how='mean') #or
    # ts_mean = ts.resample('1D').mean()
    ts_mean = ts.resample(resampling_time, how=resampling_method).ffill() # propage last valid obs forward

    # save
    ts_mean.to_csv(out_fname)
    print ('Progress --> Output creatad for observed file at %s' % out_fname)
    return {'success': True, 'output_file': out_fname}


def downloadanddailyusgsdischarge(USGS_Gage, begin_date='10/01/2010', end_date='12/30/2010',
                                     out_fname='q_obs_cfs.txt', output_unit='cfs' ):
    """
    Downloads, and then resamples the discharge data from USGS using the url of the format:
    http://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&format=rdb&site_no=10109000&period=&begin_date=2015-10-01&end_date=2015-10-31
    INPUT:
    USGS_Gage :     string, e.g. 10109000
    begin_date=     string, e.g. '10/01/2010'
    end_date=       string, e.g. ''12/30/2010'
    out_fname=      string, e.g. 'Q_cfs.txt'
    output_unit=    string, e.g. 'cfs' or 'cumecs'
    resampling_time=  string, e.g. '1D'
    resampling_method=string, e.g.'mean'
    """

    print ('Input begin date', begin_date)
    begin_date = change_date_from_mmddyyyy_to_yyyyddmm(in_date=begin_date)
    end_date = change_date_from_mmddyyyy_to_yyyyddmm(in_date=end_date)
    print ('Edited begin date', begin_date)
    print ('Required format is yyyy-mm-dd')

    urlString3 = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=%s&referred_module=sw&period=&begin_date=%s&end_date=%s' % (
    USGS_Gage, begin_date, end_date)


    response = urllib.request.urlopen(urlString3)  # instance of the file from the URL
    html = response.read()  # reads the texts into the variable html
    print ('Progress --> Observed discharge data read from the url: ', urlString3)

    with open('Q_raw.txt', 'wb') as f:
        f.write(html)

    rows_to_skip = return_data_line(afile='Q_raw.txt')
    df = pd.read_csv('Q_raw.txt', delimiter='\t', skiprows=rows_to_skip,
                     names=['agency_cd', 'USGS_Station_no', 'datatime', 'Q_cfs', 'Quality'])

    # convert datetime from string to datetime
    df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2], errors='ignore')

    # create a different dataframe with just the values and datetime
    df_datetime_val = df[['datatime', 'Q_cfs']]

    # convert the values to series
    values = []
    dates = []

    # add values to the list a
    multiplier = 1.0
    for v in df_datetime_val.iloc[:, 1]:

        if output_unit.lower() == 'cumecs' or output_unit.lower()=='cumec':
            multiplier = 0.028316846592

        values.append(round(float(v) * multiplier, 6))

    # add datatime to list b
    for v in df_datetime_val.iloc[:, 0]:
        dates.append(v)

    # prepare a panda series
    ts = pd.Series(values, index=dates)

    # save
    ts.to_csv(out_fname)
    print ('Progress --> Output creatad for observed file at %s' % out_fname)
    return {'success': True, 'output_file': out_fname}



def pull_from_hydroshare(hs_resource_id=None, output_folder=None, hs_usr_name='prasanna_310',
                         hs_password='Hydrology12!@'):
    """
    :param hs_resource_id:      hydroshare resource id for public dataset(??), that contains a single shapefile
    :return: hs_to_shp:         {'outshp_path': path of shapefile (point or polygon) based on hs_resource_id, 'error':}
    """
    from hs_restclient import HydroShare, HydroShareAuthBasic
    hs_resource_id = str(hs_resource_id)
    # :TODO this password and user name should be User's
    auth = HydroShareAuthBasic(username=hs_usr_name, password=hs_password)
    hs = HydroShare(auth=auth)

    contains_pytopkapi_file = False
    pytopkapi_files = {}
    data_folder = os.path.join(output_folder, hs_resource_id, hs_resource_id, 'data', 'contents')

    resource = hs.getResource(pid=hs_resource_id, destination=output_folder, unzip=True)

    files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    shp = [os.path.join(data_folder, f) for f in files if f.endswith('.shp')]
    tiff = [os.path.join(data_folder, f) for f in files if f.endswith('.tif')]

    ini = [f for f in files if f.endswith('.ini')]
    dat = [f for f in files if f.endswith('.dat')]
    h5 = [f for f in files if f.endswith('.h5')]

    if ('TOPKAPI.ini' in ini) and ('cell_param.dat' and 'global_param.dat' in dat) and (
        'rainfields.h5' and 'ET.h5' in h5):
        contains_pytopkapi_file = True
        pytopkapi_files = files

    return_dict = {'hs_res_id': resource, 'files': files, 'shp': shp, 'tiffs': tiff,
                   'contains_pytopkapi_file': contains_pytopkapi_file, 'pytopkapi_files': pytopkapi_files}

    return return_dict


def pull_one_file_from_hydroshare(hs_id, fname, output_folder='.', hs_usr_name='prasanna_310',
                                  hs_password='Hydrology12!@'):
    from hs_restclient import HydroShare, HydroShareAuthBasic
    auth = HydroShareAuthBasic(username=hs_usr_name, password=hs_password)
    hs = HydroShare(auth=auth)

    fpath = hs.getResourceFile(hs_id, fname, destination=output_folder)

    print ('The file downloaded from HydroShare is ', fpath)

    return fpath


def push_to_hydroshare(simulation_name=None, data_folder=None, hs_usr_name='prasanna_310', hs_password='Hydrology12!@'):
    # sys.path.append('/home/prasanna/Documents/hydroshare-jupyterhub-master/notebooks/utilities')
    print ('Progress --> Pushing files to HydroShare. This could take while...')
    from hs_restclient import HydroShare, HydroShareAuthBasic

    auth = HydroShareAuthBasic(username=hs_usr_name, password=hs_password)
    hs = HydroShare(auth=auth)

    abstract = 'This is model files for PyTOPKAPI simulation of '  # abstract for the new resource
    title = 'Model files for PyTOPKAPI simulation of ' + simulation_name  # title for the new resource
    keywords = ['PyTOPKPAI', 'Hydrologic_modeling', 'USU']  # keywords for the new resource
    rtype = 'GenericResource'  # Hydroshare resource type
    files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if
             os.path.isfile(os.path.join(data_folder, f))]

    hs_res_id_created = hs.createResource(resource_type=rtype, title=title, resource_file=files[0],
                                          resource_filename=os.path.basename(files[0]),
                                          abstract=abstract, keywords=keywords,
                                          edit_users=None, view_users=None, edit_groups=None, view_groups=None,
                                          metadata=None, extra_metadata=None, progress_callback=None)
    print ('Resources created is ', hs_res_id_created)

    for file in files[1:]:
        var2 = hs.addResourceFile(hs_res_id_created, resource_file=file, resource_filename=os.path.basename(file),
                                  progress_callback=None)
        # print ('Resources created to which file %s added is %s', (file ,hs_res_id_created))
    print ('Progress --> Successfully pushed files to HydroShare. Created HS_res_ID ', hs_res_id_created)
    return hs_res_id_created


def replace_file_in_hydroshare(existing_hs_id, data_folder=None, list_of_files_to_update=None,
                               hs_usr_name='prasanna_310', hs_password='Hydrology12!@'):
    from hs_restclient import HydroShare, HydroShareAuthBasic
    auth = HydroShareAuthBasic(username=hs_usr_name, password=hs_password)
    hs = HydroShare(auth=auth)

    for fname in list_of_files_to_update:
        fpath = os.path.join(data_folder, fname)

        try:
            # delete files from resource
            resource_id = hs.deleteResourceFile(existing_hs_id, fname)

            # add files to replace the
            resource_id = hs.addResourceFile(existing_hs_id, fpath)

        except:
            pass

    return existing_hs_id


def read_hydrograph(input_q):
    # :TODO for simiulation, read_hydrogrpah WORKS, because it skips last line. DOES NOT WORK FOR observed_q
    import pandas as pd

    f = open(input_q, "r")
    str_to_save = f.read().replace('-', ",")
    str_to_save = str_to_save.replace('\t', ",")
    str_to_save = str_to_save.replace(r"\s+", ",")

    f.close()

    # save it again
    f = open(input_q, "w")
    f.write(str_to_save)
    f.close()

    pd_ar = pd.read_csv(input_q)  #
    f = np.array(pd_ar)

    # f = np.loadtxt(input_q)  #, delimiter=",")
    # f = np.genfromtxt(input_q,  dtype=[int,int,int,int,int, float])

    q = f[:, -1]

    return q


def add_an_element_to_json(json_file, elementName, elementValue):
    with open(json_file, 'r') as oldfile:
        old_data = json.load(oldfile)
    with open(json_file, 'w') as updatefile:
        old_data[elementName] = elementValue
        json.dump(old_data, updatefile, indent=4)


def runtopnet(user_name, simulation_name, simulation_start_date, simulation_end_date, USGS_gage, timestep, threshold,
              overland_manning_fname,
              hillslope_fname, dem_fname, channel_network_fname, mask_fname, flowdir_fname,
              pore_size_dist_fname, bubbling_pressure_fname, resid_moisture_content_fname, sat_moisture_content_fname,
              conductivity_fname, soil_depth_fname,
              output_hs_rs_id_txt='topknet_model_files_metadata.txt', output_q_sim_txt='q_sim_cfs.txt'):
    """
    :param inputs as string:  Dictionary. Inputs from Tethys, or user requesting the service
    :return: Timeseries file- hydrograph, or list of input files if the user only wants input files
    """

    import pytopkapi
    from pytopkapi.results_analysis import plot_Qsim_Qobs_Rain, plot_soil_moisture_maps
    import pytopkapi.utils as ut

    simulation_folder = generate_uuid_file_path();
    print ("Progress -->  Working folder created")
    os.chdir(simulation_folder)

    # STEP1:
    run_1 = pytopkapi_run_instance(user_name=user_name,
                                   simulation_name=simulation_name,
                                   simulation_start_date=simulation_start_date,
                                   simulation_end_date=simulation_end_date,
                                   USGS_gage=USGS_gage,
                                   timestep=timestep,
                                   threshold=threshold,

                                   mask_fname=mask_fname, simulation_folder=simulation_folder,
                                   # channel_manning_fname=channel_manning_fname,
                                   overland_manning_fname=overland_manning_fname,
                                   hillslope_fname=hillslope_fname, dem_fname=dem_fname,
                                   channel_network_fname=channel_network_fname,
                                   flowdir_fname=flowdir_fname, pore_size_dist_fname=pore_size_dist_fname,
                                   bubbling_pressure_fname=bubbling_pressure_fname,
                                   resid_moisture_content_fname=resid_moisture_content_fname,
                                   sat_moisture_content_fname=sat_moisture_content_fname,
                                   conductivity_fname=conductivity_fname, soil_depth_fname=soil_depth_fname)
    # rain_fname=rain_fname, et_fname=et_fname, runoff_fname=runoff_fname)
    print ("Progress -->  Run instance created")

    # # # STEP3a: get the TIFF files, a
    raster_file_dict = run_1.raster_file_dict

    # # STEP2: Create forcing files
    abstract_pcp(startDate=simulation_start_date, endDate=simulation_end_date, cell_size=run_1.cell_size,
                 input_watershed=run_1.mask_fname, climate_variable='prcp', output_nc_fname='ppt.nc')
    str1 = 'python3.4 /home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/create_hdf5_rain.py '
    str2 = '-i ppt.nc -m ' + mask_fname + ' -o ' + simulation_folder + ' -t %s' % timestep
    os.system(str1 + str2)
    # create_pytopkapi_hdf5_from_nc(nc_f='ppt.nc', mask_tiff=mask_fname, output_folder=simulation_folder)
    print ('Progress --> Forcing files created')

    # # STEP3b get forcing files dictionary
    # # Update files
    # rain_fname = '/home/ahmet/ciwater/usu_data_service/workspace/73a3da5399944413b46b0ce7d882fa8c/rainfields.h5'
    # et_fname = '/home/ahmet/ciwater/usu_data_service/workspace/73a3da5399944413b46b0ce7d882fa8c/ET.h5'
    # runoff_fname = None
    # run_1.rain_fname =  rain_fname
    # run_1.et_fname = et_fname
    # run_1.forcing_files_dict =  {
    #                           'rain_fname':  rain_fname,
    #                           'et_fname':  et_fname  }

    pytopkapi_input_files = run_1.prepare_supporting_ini(raster_file_dict=raster_file_dict)
    print ("Progress -->  Supporting ini files created")

    # # STEP4: Run the model
    outlet_ID, no_of_cell = get_outletID_noOfCell(pytopkapi_input_files['cell_param'])
    running_instance = run_1.run_the_model(topkapi_ini=pytopkapi_input_files['TOPKAPI_ini'])
    # if running_instance['simulation_run'] == True:
    #     file_out =running_instance['results']
    print ("Progress -->  Simulation Successfully ran")

    # # # STEP5 get just the simulated hydrograph
    # ndar_Qc_out = ut.read_one_array_hdf(running_instance['results'], 'Channel', 'Qc_out')

    # # # STEP5: Retreive simulated and observed hydrograph
    # ar_date, ar_Qsim, error_checking_param = run_1.get_hydrograph()
    # print (ar_Qsim)
    # # # STEP5: Alternative
    q_sim = os.path.join(os.path.split(mask_fname)[0],
                         output_q_sim_txt)  # output_q_sim_txt # os.path.join(simulation_folder,output_q_sim_txt )
    str1 = 'python3.4 /home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/read_result_hydrograph.py '
    str2 = '-i %s -oid %s -d %s -t %s -oq %s ' % (
    run_1.path_to_results, str(outlet_ID), simulation_start_date, str(timestep), q_sim)
    os.system(str1 + str2)
    print ('Progress --> Hydrograph results Read')

    # # # STEP6 Soil Moisture Maps
    # # plot_soil_moisture_maps.run(os.path.join(simulation_folder, pytopkapi_input_files['plot_soil_moisture_ini']))
    # plot_soil_moisture_maps.run( pytopkapi_input_files['plot_soil_moisture_ini'])


    # :TODO A function to change the ini files, from absolute path to relative


    # # STEP7 Copy PyTOPKAPI files to user directory,so that the directory contains all files necessary for model run
    import shutil
    # from distutils.dir_util import copy_tree
    # copy_tree('/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/pytopkapi_for_users/',
    #           simulation_folder)

    # zip directory that are not file
    all_dirs = [os.path.join(simulation_folder, o) for o in os.listdir(simulation_folder) if
                os.path.isdir(os.path.join(simulation_folder, o))]
    for dir in all_dirs:
        output_zip_name = simulation_folder + '/' + os.path.basename(dir) + '.zip'
        zip_dir(dir, path_file_zip=output_zip_name)
        shutil.rmtree(dir, ignore_errors=True)

    print ('Progress --> Folders Zipped')

    change_ini_path_to_local(folders_with_inis=simulation_folder, string_to_change_in_files=simulation_folder + '/')

    # copy mask.tif, and the two text files
    shutil.copy2(mask_fname, simulation_folder)
    output_q_sim_txt = os.path.join(os.path.split(mask_fname)[0], output_q_sim_txt)
    shutil.copy2(output_q_sim_txt, simulation_folder)

    # upload the directory to HydroShare
    print ('Progress --> Pushing files to HydroShare. This could take while...')
    hs_res_id_created = push_to_hydroshare(simulation_name=simulation_name, data_folder=simulation_folder)
    print ('Progress --> Successfully pushed files to HydroShare. Created HS_res_ID ', hs_res_id_created)

    # # # LAST STEP: zip the diretory to return to user. Delete the working directory
    # print ('Trying to zip everything now')
    # zip_dir(simulation_folder, output_zip)

    with open(output_hs_rs_id_txt, 'w') as f:
        f.write(str(hs_res_id_created))

    # copy created
    shutil.copy2(os.path.join(os.path.split(mask_fname)[0], output_hs_rs_id_txt), simulation_folder)

    print ('deleting working directory now')
    delete_working_uuid_directory(simulation_folder)

    return {'success': 'True'}


class pytopkapi_run_instance:
    def __init__(self, user_name=None, simulation_name=None, simulation_start_date=None, simulation_end_date=None,
                 USGS_gage=None, timestep=None, threshold=None, mask_fname=None, simulation_folder=None,
                 channel_manning_fname=None, overland_manning_fname=None,
                 hillslope_fname=None, dem_fname=None, channel_network_fname=None, flowdir_fname=None,
                 pore_size_dist_fname=None, bubbling_pressure_fname=None, resid_moisture_content_fname=None,
                 sat_moisture_content_fname=None, conductivity_fname=None, soil_depth_fname=None,
                 rain_fname=None, et_fname=None, runoff_fname=None, modifying=False):

        self.simulation_name = simulation_name
        self.user_name = user_name
        self.simulation_folder = simulation_folder  # :TODO check the requirements
        self.simulation_start_date = simulation_start_date  # format: mm/dd/yyyy
        self.simulation_end_date = simulation_end_date

        self.timestep = timestep
        self.USGS_gage = str(USGS_gage)
        self.threshold = threshold

        # files
        self.channel_manning_fname = channel_manning_fname
        self.overland_manning_fname = overland_manning_fname
        self.hillslope_fname = hillslope_fname
        self.dem_fname = dem_fname
        self.channel_network_fname = channel_network_fname
        self.mask_fname = mask_fname
        self.flowdir_fname = flowdir_fname
        self.pore_size_dist_fname = pore_size_dist_fname
        self.bubbling_pressure_fname = bubbling_pressure_fname
        self.resid_moisture_content_fname = resid_moisture_content_fname
        self.sat_moisture_content_fname = sat_moisture_content_fname
        self.conductivity_fname = conductivity_fname
        self.soil_depth_fname = soil_depth_fname
        self.rain_fname = rain_fname
        self.et_fname = et_fname           # netcdf
        self.runoff_fname = runoff_fname   # netcdf

        self.raster_file_dict = {'hillslope_fname': self.hillslope_fname,
                                 'pore_size_dist_fname': self.pore_size_dist_fname,
                                 'channel_network_fname': self.channel_network_fname,
                                 'bubbling_pressure_fname': self.bubbling_pressure_fname,
                                 'sat_moisture_content_fname': self.sat_moisture_content_fname,
                                 'conductivity_fname': self.conductivity_fname,
                                 'dem_fname': self.dem_fname,
                                 'resid_moisture_content_fname': self.resid_moisture_content_fname,
                                 # 'channel_manning_fname': channel_manning_fname,
                                 'soil_depth_fname': self.soil_depth_fname,
                                 'flowdir_fname': self.flowdir_fname,
                                 'mask_fname': self.mask_fname,
                                 'overland_manning_fname': self.overland_manning_fname}

        # calculated
        self.cell_size = get_cellSize(self.mask_fname)

        # default numeric parameters
        self.pvs_t0 = 50
        self.vo_t0 = 100

        if not modifying:
            self.vo_t0 = 0.003* int(self.cell_size) ** 2  # 1% of cell volume of unit depth
        self.qc_t0 = 0.0
        self.kc = 1.0

        # default calibration parameters
        self.fac_l = 1.0
        self.fac_ks = 1.0
        self.fac_n_o = 1.0
        self.fac_n_c = 1.0
        self.fac_th_s = 1.0

        # soil map parameters
        self.soil_variable = 4  # 4: moisture map
        # self.end_timestep = 10





        # global names (NOT THE FULL PATH)
        self.cell_param_name = 'cell_param.dat'
        self.cell_param_unmodified_name = 'cell_param_unmodified.dat'
        self.global_param_name = 'global_param.dat'
        self.create_file_name = 'create_file.ini'
        self.TOPKAPI_ini_name = 'TOPKAPI.ini'
        self.zero_slope_name = 'zero_slope_management.ini'
        self.plot_flow_name = 'plot-flow-precip.ini'
        self.plot_soil_moisture_name = 'plot-soil-moisture-maps.ini'
        self.results_h5_name = 'results.h5'
        self.run_detail_JSON = 'run_info.txt'
        self.q_obs_txt = 'q_obs_cfs.txt'
        self.q_sim_txt = 'q_sim_cfs.txt'

        # some default paths..
        self.path_to_rain_txt_file = os.path.join(self.simulation_folder, 'Rainfall.txt')
        self.path_to_ET_txt_file = os.path.join(self.simulation_folder, 'ET.txt')

        self.path_to_change_result_log = os.path.join(self.simulation_folder, 'change_result_log.dat')

        self.path_to_results = os.path.join(self.simulation_folder, self.results_h5_name)

        # if self.rain_fname == None:
        #     self.rain_h5_name = 'rainfields.h5'
        #     self.rain_fname = os.path.join(self.simulation_folder,  self.rain_h5_name )
        # if self.et_fname == None:
        #     self.et_h5_name = 'ET.h5'
        #     self.et_fname = os.path.join(self.simulation_folder, self.et_h5_name)
        # if self.runoff_fname == None:
        #     self.runoff_fname = os.path.join(self.simulation_folder, 'Runoff.dat')

        self.forcing_files_dict = {
            'rain_fname': os.path.join(self.simulation_folder, 'rainfields.h5'),
            'et_fname': os.path.join(self.simulation_folder, 'ET.h5')}

        if not modifying:
            self.write_run_detail(first_time=True)
        # create ini files
        print ("Progress -->  Run instance created")

        return

    # :todo PREPARING TIFF FILES. But for the first part of project, already created files will be used
    def prepare_tiffs(self):
        # download DEM, NLCD
        # path to LUT, defaulted
        # soil querries etc.
        pass

    def write_run_detail(self, run_detail={}, first_time=False):
        # call this functino as soon as run instance is created, so that empty list is there for runJSON['run'] element
        # for any run_detail given (apart from the time of initialization), the simulated discharge is written to the run element
        '''
        :param run_detail:  { 'numeric_parameter'={}, 'calib_parameter'={}, 'q_obs':[], 'q_sim':[], 'nash':.90}
        :return: JSON file containing detail about model created, and the different runs acted on it
        '''

        if first_time:  # i.e. called as sooon as run instance created
            runJSON = {}
            runJSON['user_name'] = self.user_name
            runJSON['simulation_name'] = self.simulation_name
            runJSON['simulation_start_date'] = self.simulation_start_date
            runJSON['simulation_end_date'] = self.simulation_end_date
            runJSON['timestep'] = self.timestep
            runJSON['USGS_gage'] = self.USGS_gage
            runJSON['threshold'] = self.threshold
            runJSON['cell_size'] = self.cell_size
            runJSON['runs'] = []  # a list of run_detail dictionary

            # save
            with open(self.run_detail_JSON, 'w') as newfile:
                json.dump(runJSON, newfile)
        else:
            # json cannot append. So, step 1) read everything, step 2) Make modification. step 3) Save.
            with open(self.run_detail_JSON, 'r') as oldfile:
                old_data = json.load(oldfile)
                current_timestep_date = startDate = datetime.strptime(old_data['simulation_start_date'], '%m/%d/%Y')
                timestep = int(old_data['timestep'])


            # 1) read hydrograph from file
            discharge_list = read_hydrograph(os.path.join(self.simulation_folder, 'q_sim_cfs.txt'))

            # 2) create yr_mon_day_hr_min_discharge_list
            yr_mon_day_hr_min_discharge_list = []
            for value in discharge_list:

                yr_mon_day_hr_min_discharge_list.append(
                    [current_timestep_date.year, current_timestep_date.month, current_timestep_date.day,
                     current_timestep_date.hour, current_timestep_date.minute, float(value)])

                current_timestep_date = current_timestep_date + timedelta(hours=timestep)

            run_detail['simulated_discharge'] = yr_mon_day_hr_min_discharge_list
            old_data['runs'].append(run_detail)

            with open(self.run_detail_JSON, 'w') as updatefile:
                json.dump(old_data, updatefile)

        return {'success': True}

    # :todo PREPARING FORCING FILES.

    def prepare_forcing_files(self, timeseries_source, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.timeseries_source = timeseries_source

        # if certain source, do something. Else do other things

        #
        pass

    def prepare_supporting_ini(self, raster_file_dict):
        from pytopkapi.parameter_utils.create_file import generate_param_file
        from pytopkapi.parameter_utils import modify_file

        # FILE-1: global_parameter.dat
        path_to_global_param = create_global_param(
            path_to_out_dat=os.path.join(self.simulation_folder, self.global_param_name),
            A_thres=float(self.threshold * 1000000), X=float(self.cell_size),
            Dt=float(self.timestep) * 3600)

        # FILE-2:  create_file.ini
        path_to_create_file_ini = create_config_files_create_file(
            path_to_in_cell_param_dat=os.path.join(self.simulation_folder, self.cell_param_unmodified_name),
            path_to_out_ini=os.path.join(self.simulation_folder, self.create_file_name),
            raster_files_dict=raster_file_dict)

        # FILE-3: create cell_param_modified.dat
        generate_param_file(path_to_create_file_ini, isolated_cells=False)
        print ("Progress --> Cell Parameter file created")

        # FILE-4: create zero_slope_management.ini
        path_to_zero_slope_ini = create_config_files_zero_slope_mngmt(
            path_to_unmodified_cell_param_dat=os.path.join(self.simulation_folder, self.cell_param_unmodified_name),
            path_to_modified_cell_param_dat=os.path.join(self.simulation_folder, self.cell_param_name),
            path_to_out_ini=os.path.join(self.simulation_folder, self.zero_slope_name),
            cell_size=self.cell_size)

        # FILE-5: create cell_param.dat, after zero slope management correction
        modify_file.zero_slope_management(path_to_zero_slope_ini)
        print ("Progress -->  Zero Slope corrections made")

        # TOPKAPI.ini, :TODO result is output, so think. Also, rainfields,ET...
        input_files = {'file_global_param': os.path.join(self.simulation_folder, self.global_param_name),
                       'file_cell_param': os.path.join(self.simulation_folder, self.cell_param_name),
                       'results': self.path_to_results,
                       'file_rain': self.forcing_files_dict['rain_fname'],
                       'file_et': self.forcing_files_dict['et_fname'],
                       'soil_maps_folder': os.path.join(self.simulation_folder, 'soil_maps'),
                       }
        output_files = {'file_out': self.path_to_results,
                        'file_change_log_out': self.path_to_change_result_log,
                        'append_output': 'False'}

        path_to_TOPKAPI_ini = create_config_files_TOPKAPI_ini(
            path_to_out_ini=os.path.join(self.simulation_folder, self.TOPKAPI_ini_name),
            input_files=input_files, output_files=output_files)

        # plot-flow-precip.ini
        files_for_plot = {'path_to_results': self.path_to_results,
                          'path_to_runoff_file': self.runoff_fname,
                          'path_to_rain': self.forcing_files_dict['rain_fname'],
                          }

        # self.end_timestep = self.get_rainfall_end_timestep()
        end_timestep = (
        datetime.strptime(self.simulation_end_date, "%m/%d/%Y") - datetime.strptime(self.simulation_start_date,
                                                                                    "%m/%d/%Y")).days
        self.outlet_ID, self.no_of_cell = get_outletID_noOfCell(
            os.path.join(self.simulation_folder, self.cell_param_name))
        parameters_for_plot = {'outlet_ID': self.outlet_ID, 'calibration_start_date': self.simulation_start_date,
                               'start_timestep': 1, 'end_timestep': abs(end_timestep),
                               'variable_to_plot_maps': self.soil_variable}  # incl. soil moisture maps

        path_to_config_files_plot_flow_precip_ini = create_config_files_plot_flow_precip(
            path_to_out_ini=os.path.join(self.simulation_folder, self.plot_flow_name),
            files=files_for_plot,
            parameters=parameters_for_plot)

        # create ini: plot-soil-moisture-map.ini
        create_config_files_plot_soil_moisture_map(os.path.join(self.simulation_folder, self.plot_soil_moisture_name),
                                                   input_files,
                                                   parameters_for_plot)

        pytopkapi_input_files = {}

        for a_file in os.listdir(os.getcwd()):  # self.simulation_folder):
            if a_file == self.cell_param_name:
                pytopkapi_input_files['cell_param'] = os.path.join(self.simulation_folder, a_file)
            if a_file == self.global_param_name:
                pytopkapi_input_files['global_param'] = os.path.join(self.simulation_folder, a_file)
            if a_file == self.TOPKAPI_ini_name:
                pytopkapi_input_files['TOPKAPI_ini'] = os.path.join(self.simulation_folder, a_file)

            if a_file == self.create_file_name:
                pytopkapi_input_files['create_file_ini'] = os.path.join(self.simulation_folder, a_file)
            if a_file == self.zero_slope_name:
                pytopkapi_input_files['zero_slope_ini'] = os.path.join(self.simulation_folder, a_file)
            if a_file == self.plot_flow_name:
                pytopkapi_input_files['plot_flow_ini'] = os.path.join(self.simulation_folder, a_file)
            if a_file == self.plot_soil_moisture_name:
                pytopkapi_input_files['plot_soil_moisture_ini'] = os.path.join(self.simulation_folder, a_file)

            # if a_file == self.rain_h5_name:
            #     pytopkapi_input_files['file_rain'] = os.path.join(self.simulation_folder, a_file)
            # if a_file == self.et_h5_name:
            #     pytopkapi_input_files['file_et'] = os.path.join(self.simulation_folder, a_file)
            if a_file == self.results_h5_name:
                pytopkapi_input_files['results'] = os.path.join(self.simulation_folder, a_file)

        # pytopkapi_input_files = {
        #     'cell_param':'',
        #     'global_param':'',
        #     'TOPKAPI_ini':'',
        #     'create_file_ini': '',
        #     'zero_slope_ini': '',
        #     'plot_flow_ini': '',
        #     'plot_soil_moisture_ini': '',
        #     'file_rain': '',
        #     'file_et': '',
        #     'results': '',
        #
        # }

        pytopkapi_input_files['file_rain'] = self.rain_fname
        pytopkapi_input_files['file_et'] = self.et_fname

        print ("Progress -->  Supporting ini files created")
        return pytopkapi_input_files

    def run_backup(self, calibration_parameters="", numerical_values=""):
        """
        Parameters
        ----------
        calibration_parameters: a list, of calibration parameters. [fac_l, fac_ks, fac_n_o, fac_n_c, fac_th_s]
        initial_numerical_values: a list, of initial numeric values [pvs_t0,vo_t0,qc_t0, kc  ]
        simulation_folder: folder where results and everything are there

        Returns: a list. [Runname, date_time, nash value,\t, [Q_sim]]
        -------
        """

        self.numerical_values = numerical_values

        # these values are already there in the init, so not necessary to input them to run the model
        if calibration_parameters != "":
            self.fac_l = calibration_parameters[0]
            self.fac_ks = calibration_parameters[1]
            self.fac_n_o = calibration_parameters[2]
            self.fac_n_c = calibration_parameters[3]
            self.fac_th_s = calibration_parameters[4]

        create_config_files_TOPKAPI_ini(self.simulation_folder, 'False', 'False', self.fac_l, self.fac_ks,
                                        self.fac_n_o, self.fac_n_c, self.fac_th_s)

        # make changes to numerical value section in cell_param.dat
        if numerical_values != "":
            self.cell_param_array = np.genfromtxt(self.simulation_folder + '/cell_param.dat', delimiter=' ')

            self.pvs_t0 = numerical_values[0]  # located at col-16 (15 in index)
            self.vo_t0 = numerical_values[1]  # located at col-17 (16 in index)
            self.qc_t0 = numerical_values[2]  # located at col-18 (17 in index)
            self.kc = numerical_values[3]  # located at col-19 (18 in index)

            # change the values that exist
            self.cell_param_array[:, 15] = np.zeros((int(self.cell_param_array[:, 15].shape[0]),)) + self.pvs_t0
            self.cell_param_array[:, 16] = np.zeros((int(self.cell_param_array[:, 16].shape[0]),)) + self.vo_t0
            self.cell_param_array[:, 17] = np.zeros((int(self.cell_param_array[:, 17].shape[0]),)) + self.qc_t0
            self.cell_param_array[:, 18] = np.zeros((int(self.cell_param_array[:, 18].shape[0]),)) + self.kc

            np.savetxt(self.simulation_folder + '/cell_param.dat', self.cell_param_array, delimiter=' ')

        self.outlet_ID, self.no_of_cell = get_outletID_noOfCell(self.simulation_folder + '/cell_param.dat')

        # run the program now
        pytopkapi.run(self.simulation_folder + '/TOPKAPI.ini')

        return "successfully ran the simulation"

    def run_the_model(self, topkapi_ini=None,
                      new_calib_param={},
                      new_numeric_param={}, cell_param_file=''):
        """
        an be used to run the model for the first time, or to re-run making modifications

        Parameters
        ----------
        calib_param:    {'fac_l': 1.,'fac_ks': 1., 'fac_n_o': 1., 'fac_n_c': 1., 'fac_th_s': 1}
        numeric_param:  {'pvs_t0': 90., 'vo_t0': 1000.0, 'qc_t0': 0, 'kc': 1.0}
        simulation_folder: folder where results and everything are there

        Returns: a list. [Runname, date_time, nash value,\t, [Q_sim]]
        -------
        """

        if topkapi_ini == None:
            topkapi_ini = os.path.join(self.simulation_folder, self.TOPKAPI_ini_name)

        # read the topkapi_ini_file to get the input_files, numeric values etc.
        # these are old values. If they are not substituted, the old values will be used to create new param file

        config = SafeConfigParser()
        config.read(topkapi_ini)

        file_global_param = config.get('input_files', 'file_global_param')
        file_cell_param = config.get('input_files', 'file_cell_param')
        file_rain = config.get('input_files', 'file_rain')
        file_et = config.get('input_files', 'file_et')

        file_out = config.get('output_files', 'file_out')
        file_change_log_out = config.get('output_files', 'file_change_log_out')
        append_output = config.get('output_files', 'append_output')

        external_flow = config.get('external_flow', 'external_flow')

        solve_s = config.get('numerical_options', 'solve_s')
        solve_o = config.get('numerical_options', 'solve_o')
        solve_c = config.get('numerical_options', 'solve_c')
        only_channel_output = config.get('numerical_options', 'only_channel_output')

        fac_l = config.get('calib_params', 'fac_l')
        fac_ks = config.get('calib_params', 'fac_ks')
        fac_n_o = config.get('calib_params', 'fac_n_o')
        fac_n_c = config.get('calib_params', 'fac_n_c')
        fac_th_s = config.get('calib_params', 'fac_th_s')

        old_input_files = {'file_global_param': file_global_param, 'file_cell_param': file_cell_param,
                           'results': file_out, 'file_rain': file_rain, 'file_et': file_et}
        old_output_files = {'file_out': file_out, 'file_change_log_out': file_change_log_out,
                            'append_output': append_output}
        old_calib_param = {'fac_l': fac_l, 'fac_ks': fac_ks, 'fac_n_o': fac_n_o, 'fac_n_c': fac_n_c,
                           'fac_th_s': fac_th_s}

        # new values!!!
        # calib_parameters:
        if new_calib_param != {}:
            # fac_l = new_calib_param['fac_l']
            # fac_ks = new_calib_param['fac_ks']
            # fac_n_o = new_calib_param['fac_n_o']
            # fac_n_c = new_calib_param['fac_n_c']
            # fac_th_s = new_calib_param['fac_th_s']
            # new_calib_param = {'fac_l': fac_l, 'fac_ks': fac_ks, 'fac_n_o': fac_n_o, 'fac_n_c': fac_n_c, 'fac_th_s': fac_th_s}

            # we dont change input and output file location. We only change calib_param
            rename_topkapi_ini = topkapi_ini.replace('.ini', '_old.ini')
            os.rename(topkapi_ini, rename_topkapi_ini)
            # create new one, but maintain the old name
            create_config_files_TOPKAPI_ini(input_files=old_input_files, output_files=old_output_files,
                                            calib_param=new_calib_param, path_to_out_ini=topkapi_ini)

        # make changes to numerical value section in cell_param.dat
        if new_numeric_param != {}:
            cell_param_array = np.genfromtxt(cell_param_file, delimiter=' ')

            pvs_t0 = new_numeric_param['pvs_t0']  # located at col-16 (15 in index)
            vo_t0 = new_numeric_param['vo_t0']  # located at col-17 (16 in index)
            qc_t0 = new_numeric_param['qc_t0']  # located at col-18 (17 in index)
            kc = new_numeric_param['kc']  # located at col-19 (18 in index)

            # change the values that exist
            cell_param_array[:, 15] = np.zeros((int(cell_param_array[:, 15].shape[0]),)) + pvs_t0
            cell_param_array[:, 16] = np.zeros((int(cell_param_array[:, 16].shape[0]),)) + vo_t0
            cell_param_array[:, 17] = np.zeros((int(cell_param_array[:, 17].shape[0]),)) + qc_t0
            cell_param_array[:, 18] = np.zeros((int(cell_param_array[:, 18].shape[0]),)) + kc

            np.savetxt(file_cell_param, cell_param_array, delimiter=' ')  # TODO: this may not work, Output file..

        # run the program now
        pytopkapi.run(topkapi_ini)  # :TODO use try except to catch the error

        if new_calib_param == {}:  # FIRST RUN
            new_calib_param = {'fac_l': self.fac_l, 'fac_ks': self.fac_ks, 'fac_n_o': self.fac_n_o,
                               'fac_n_c': self.fac_n_c, 'fac_th_s': self.fac_th_s}

        if new_numeric_param == {}:  # FIRST RUN
            new_numeric_param = {'pvs_t0': self.pvs_t0, 'vo_t0': self.vo_t0, 'qc_t0': self.qc_t0, 'kc': self.kc}

        # get the
        run_detail = {'numeric_param': new_numeric_param, 'calib_parameter': new_calib_param}

        # # write to the run info json file
        # self.write_run_detail(run_detail=run_detail)

        print ("Progress -->  Simulation Successfully ran")
        return run_detail  # {'simulation_run':True, 'results':file_out}

    def get_rainfall_end_timestep(self):
        h5file = h5py.File(self.forcing_files_dict['rain_fname'])

        dset_string = '/sample_event/rainfall'
        ndar_rain = h5file[dset_string][...]

        end_timestep = ndar_rain.shape[0]

        h5file.close()
        return end_timestep

    def get_Qsim_and_error(self, TOPKAPI_ini):

        path_to_results = get_variable_in_file_config_parser(section='output_files', variable='file_out',
                                                             ini_file=TOPKAPI_ini)
        path_to_cell_param = get_variable_in_file_config_parser(section='input_files', variable='file_cell_param',
                                                                ini_file=TOPKAPI_ini)

        self.outlet_ID, self.no_of_cell = get_outletID_noOfCell(path_to_cell_param)
        self.date, self.ar_Q_sim, self.error_checking_param = get_hydrograph(path_to_results, self.path_to_Q_observed,
                                                                             self.outlet_ID)

        print ("Date and Q-simulated array read from the prepared run")
        return self.date, self.ar_Q_sim, self.error_checking_param

    def get_hydrograph(self, file_Qsim=None, file_Qobs=None, outlet_ID=None):
        '''
        Parameters
        ----------
        image_out: fname for image of hydrographs
        file_Qobs: path to text file containing conserved value series
        outlet_ID: outlet_ID
        Returns
        -------
        list1 -> list, error checking parameters such as nash, rsme etc.
        list2 -> list, Q_simulated
        '''
        import pytopkapi
        import pytopkapi.utils as ut
        from pytopkapi.results_analysis import plot_Qsim_Qobs_Rain as pt

        from datetime import datetime

        file_Qsim = self.path_to_results
        file_Qobs = self.path_to_Q_observed
        outlet_ID = self.outlet_ID

        group_name = 'sample_event'
        Qobs = True
        Pobs = False
        nash = True

        # Read the obs
        # Qobs
        ar_date, ar_Qobs = pt.read_observed_flow(file_Qobs)

        # Read the simulated data Q
        file_h5 = file_Qsim
        ndar_Qc_out = ut.read_one_array_hdf(file_h5, 'Channel', 'Qc_out')
        ar_Qsim = ndar_Qc_out[1:, outlet_ID]

        # error estimation
        nash_value = ut.Nash(ar_Qsim, ar_Qobs)
        RMSE = ut.RMSE(ar_Qsim, ar_Qobs)
        RMSE_norm = ut.RMSE_norm(ar_Qsim, ar_Qobs)
        Bias_cumul = ut.Bias_cumul(ar_Qsim, ar_Qobs)
        Diff_cumul = ut.Diff_cumul(ar_Qsim, ar_Qobs)
        Abs_cumul = ut.Abs_cumul(ar_Qsim, ar_Qobs)
        Err_cumul = ut.Err_cumul(ar_Qsim, ar_Qobs)

        error_checking_param = [str(nash_value)[0:5], str(RMSE)[0:5], str(RMSE_norm)[0:5], str(Bias_cumul)[0:5],
                                str(Diff_cumul)[0:5], str(Abs_cumul)[0:5], str(Err_cumul)[0:5]]

        return ar_date, ar_Qsim, error_checking_param

    def get_cell_size_demo(self):

        cell_size = get_cellSize(self.simulation_folder + '/mask_r.tif')
        outlet_ID, no_of_cell = get_outletID_noOfCell(self.simulation_folder + '/cell_param.dat')
        return str(outlet_ID) + "--" + str(cell_size)

    def write_json_response(self, output_response_txt, hs_res_id_created, q_obs_status=False):
        # this function reads the two q -> obs and sim, and writes them to JSON file for output
        returnJSON = {}

        simulation_folder = os.path.split(output_response_txt)[0]
        q_obs_fname = os.path.join(self.simulation_folder, 'q_obs_cfs.txt')
        q_sim_fname = os.path.join(self.simulation_folder, 'q_sim_cfs.txt')

        returnJSON['simulated_hydrograph'] = list(
            read_hydrograph(q_sim_fname))  # list() coz np array not accepted by json
        returnJSON['hs_res_id_created'] = hs_res_id_created

        if q_obs_status == True:
            returnJSON['observed_hydrograph'] = list(read_hydrograph(q_obs_fname))

        with open(output_response_txt, 'w') as outfile:
            json.dump(returnJSON, outfile)

        return


        #  dataservices functions


def runpytopkapi(user_name, simulation_name, simulation_start_date, simulation_end_date, USGS_gage, timestep, threshold,
                 # channel_manning_fname,
                 overland_manning_fname,
                 hillslope_fname, dem_fname, channel_network_fname, mask_fname, flowdir_fname,
                 pore_size_dist_fname, bubbling_pressure_fname, resid_moisture_content_fname,
                 sat_moisture_content_fname, conductivity_fname, soil_depth_fname,
                 # rain_fname, et_fname, runoff_fname,
                 output_zip='pytopkpai_model_files.zip'):
    """
    :param inputs as string:  Dictionary. Inputs from Tethys, or user requesting the service
    :return: Timeseries file- hydrograph, or list of input files if the user only wants input files
    """

    import pytopkapi
    from pytopkapi.results_analysis import plot_Qsim_Qobs_Rain, plot_soil_moisture_maps
    import pytopkapi.utils as ut

    simulation_folder = generate_uuid_file_path();
    print ("Progress -->  Working folder created")
    os.chdir(simulation_folder)

    # STEP1: :TODO edit run instance making procedures
    run_1 = pytopkapi_run_instance(user_name=user_name,
                                   simulation_name=simulation_name,
                                   simulation_start_date=simulation_start_date,
                                   simulation_end_date=simulation_end_date,
                                   USGS_gage=USGS_gage,
                                   timestep=timestep,
                                   threshold=threshold,

                                   mask_fname=mask_fname, simulation_folder=simulation_folder,
                                   # channel_manning_fname=channel_manning_fname,
                                   overland_manning_fname=overland_manning_fname,
                                   hillslope_fname=hillslope_fname, dem_fname=dem_fname,
                                   channel_network_fname=channel_network_fname,
                                   flowdir_fname=flowdir_fname, pore_size_dist_fname=pore_size_dist_fname,
                                   bubbling_pressure_fname=bubbling_pressure_fname,
                                   resid_moisture_content_fname=resid_moisture_content_fname,
                                   sat_moisture_content_fname=sat_moisture_content_fname,
                                   conductivity_fname=conductivity_fname, soil_depth_fname=soil_depth_fname)
    # rain_fname=rain_fname, et_fname=et_fname, runoff_fname=runoff_fname)
    print ("Progress -->  Run instance created")

    # # STEP2: Create forcing files
    # forcing_files_dict = createforcingfilesforpytopkapi(mask_tiff=raster_file_dict['mask_fname'])
    abstract_pcp(startDate=simulation_start_date, endDate=simulation_end_date, cell_size=run_1.cell_size,
                 input_watershed=run_1.mask_fname, climate_variable='prcp', output_nc_fname='ppt.nc')

    create_pytopkapi_hdf5_from_nc(nc_f='ppt.nc', mask_tiff=mask_fname, output_folder=simulation_folder)
    print ('Progress --> forcing files created')

    # # Update files
    run_1.path_to_rain_h5 = os.path.join(simulation_folder, "rainfields.h5")
    run_1.path_to_et_h5 = os.path.join(simulation_folder, "ET.h5")

    # # STEP3: get the TIFF files, and forcing files dictionary
    raster_file_dict = run_1.raster_file_dict
    forcing_files_dict = run_1.forcing_files_dict

    pytopkapi_input_files = run_1.prepare_supporting_ini(raster_file_dict=raster_file_dict)

    print ("Progress -->  Supporting ini files created")

    for file in pytopkapi_input_files:
        print (file)

    # # STEP4: Run the model
    outlet_ID, no_of_cell = get_outletID_noOfCell(pytopkapi_input_files['cell_param'])
    fake_rain_h5 = create_rain_ET_from_cell_no(simulation_folder, no_of_cell)

    running_instance = run_1.run_the_model(topkapi_ini=pytopkapi_input_files['TOPKAPI_ini'])
    # if running_instance['simulation_run'] == True:
    #     file_out =running_instance['results']



    # # STEP5 get just the simulated hydrograph
    ndar_Qc_out = ut.read_one_array_hdf(running_instance['results'], 'Channel', 'Qc_out')
    ar_Qsim = ndar_Qc_out[1:, outlet_ID]
    print (ar_Qsim)

    # # # STEP5: Retreive simulated and observed hydrograph
    # ar_date, ar_Qsim, error_checking_param = run_1.get_hydrograph()
    # print (ar_Qsim)

    # # STEP6 Soil Moisture Maps
    # plot_soil_moisture_maps.run(os.path.join(simulation_folder, pytopkapi_input_files['plot_soil_moisture_ini']))
    plot_soil_moisture_maps.run(pytopkapi_input_files['plot_soil_moisture_ini'])

    # # STEP7 Copy PyTOPKAPI files to user directory,so that the directory contains all files necessary for model run
    from distutils.dir_util import copy_tree
    copy_tree('/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/pytopkapi_for_users/',
              simulation_folder)

    # # LAST STEP: zip the diretory to return to user. Delete the working directory
    zip_dir(simulation_folder, output_zip)
    delete_working_uuid_directory(simulation_folder)

    return {'success': 'True'}


def runpytopkapi2(user_name, simulation_name, simulation_start_date, simulation_end_date, USGS_gage, timestep,
                  threshold,
                  outlet_y, outlet_x, box_topY, box_bottomY, box_rightX, box_leftX, timeseries_source,
                  cell_size, model_engine,
                  output_zip='pytopkpai_model_files.zip'):
    """
    :param inputs as string:  Dictionary. Inputs from Tethys, or user requesting the service
    :return: Timeseries file- hydrograph, or list of input files if the user only wants input files
    """

    epsgcode = 102003

    simulation_folder = generate_uuid_file_path();
    print ("Progress -->  Working folder created at %s" % simulation_folder)
    # os.chdir(simulation_folder)

    # rastersubset
    step1_clip = get_raster_subset(input_raster="/home/ahmet/hydosdata/subsetsource/nedWesternUS.tif",
                                   output_raster='DEM.tif', xmin=box_leftX, ymax=box_topY, xmax=box_rightX,
                                   ymin=box_bottomY)
    print ("Progress -->  DEM was clipped")

    # projectresamplerasterepsg
    step2_resample_proj = project_and_resample_Raster_EPSG(input_raster='DEM.tif',
                                                           output_raster='DEM_prj_resampled.tif', dx=cell_size,
                                                           dy=cell_size, epsg_code=epsgcode, resample='bilinear')
    print ("Progress -->  clipped DEM was resampled and projected")

    # createoutletshapefile
    step3_def_outlet = create_OutletShape_Wrapper(outletPointX=outlet_x, outletPointY=outlet_y,
                                                  output_shape_file_name=os.getcwd() + '/outlet.shp')
    print ("Progress --> Outlet shapefile was created for the given x,y")

    # projectshapefileepsg
    step4_proj_outlet = project_shapefile_EPSG(input_shape_file=os.getcwd() + '/outlet/outlet.shp',
                                               output_shape_file='outlet_prj.shp', epsg_code=epsgcode)
    print ("Progress -->  The outlet created was projected")

    #  delineatewatershedtogetcompleterasterset
    step5_terrain_files = delineate_watershed_to_get_complete_raster_set(input_DEM_raster='DEM_prj_resampled.tif',
                                                                         stream_threshold=threshold,
                                                                         output_raster='mask.tif',
                                                                         output_outlet_shapefile='corrected_outlet.shp',
                                                                         input_outlet_shapefile='outlet_prj.shp', )
    # slope,
    step6_slope = computeRasterSlope(input_raster='fel.tif', output_raster='slope.tif')
    print ("Progress -->  The slope, in degree, was caluclated")

    # nlcd
    step7_get_nlcd = project_and_clip_raster(input_raster='/home/ahmet/hydosdata/nlcd2011CONUS/nlcd2011CONUS.tif',
                                             reference_raster='mask.tif', output_raster='nlcd.tif')
    print ("Progress -->  The NLCD file for the region was clipped")

    # nlcd to n_overland (reclassify)
    step8_nlcd2n = reclassify_raster_with_LUT(input_raster='nlcd.tif',
                                              LUT='/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/LUT_NLCD2n.csv',
                                              output_raster='n_overland.tif')
    print ("Progress -->  The NLCD was reclassified to get the Manning's n for overland")

    # 9. soil_data
    step9_soil_files = download_soil_data_for_pytopkapi('mask.tif')
    print ("Progress -->  The soil files were created after querrying NRCS database using R")

    # # 1. rastersubset
    # cmdString = "gdal_translate"+" "+"-projwin"+" "+str(box_leftX)+" "+str(box_topY)+" "+str(box_rightX)+" "+str(box_bottomY)+" "+'/home/ahmet/hydosdata/subsetsource/nedWesternUS.tif'+" "+simulation_folder+'/DEM.tif'
    # os.system(cmdString); print ("Progress -->  DEM was clipped")
    #
    # # 2. projectresamplerasterepsg
    # cmdString = "gdalwarp -t_srs EPSG:"+str(epsgcode)+" -tr "+str(cell_size)+" "+str(cell_size)+" -r "+'bilinear'+" -overwrite "+simulation_folder+'/DEM.tif'+" "+simulation_folder+'/DEM_prj_resampled.tif'
    # os.system(cmdString);print ("Progress -->  clipped DEM was resampled and projected")
    #
    # # 3. createoutletshapefile
    # create_OutletShape(simulation_folder, outlet_x, outlet_y, 'outlet.shp')
    # print ("Progress --> Outlet shapefile was created for the given x,y")
    #
    # # 4. projectshapefileepsg
    # cmdString = "ogr2ogr -t_srs EPSG:" + str(epsgcode) + " " + os.path.join(simulation_folder,'outlet_prj.shp') + " " + os.path.join(simulation_folder,'outlet.shp')
    # os.system(cmdString);print ("Progress -->  The outlet created was projected")
    #
    #
    #
    # # 5. delineatewatershedtogetcompleterasterset
    # os.chdir(simulation_folder)
    # step5_terrain_files =delineate_watershed_to_get_complete_raster_set(input_DEM_raster= 'DEM_prj_resampled.tif',
    #                                                stream_threshold=threshold, output_raster='mask.tif', output_outlet_shapefile='corrected_outlet.shp',
    #                                                 input_outlet_shapefile='outlet_prj.shp', )
    # print ("Progress -->  The terrain files were created")
    #
    #
    # # 6. slope,
    # cmdString = "gdaldem" + " " + "slope" + " " + os.path.join(simulation_folder,'fel.tif') + " " +  os.path.join(simulation_folder,'slope.tif') + " " + "-compute_edges"
    # os.system(cmdString);print ("Progress -->  The slope, in degree, was caluclated")
    #
    # # 7. nlcd
    # cmdString = "gdal_translate"+" "+"-projwin"+" "+str(box_leftX)+" "+str(box_topY)+" "+str(box_rightX)+" "+str(box_bottomY)+" "+'/home/ahmet/hydosdata/nlcd2011CONUS/nlcd2011CONUS.tif'+" "+simulation_folder+'/DEM.tif'
    # os.system(cmdString);print ("Progress -->  The NLCD file for the region was clipped")
    #
    # # 8. nlcd to n_overland (reclassify)
    # step8_nlcd2n = reclassify_raster_with_LUT( input_raster='nlcd.tif',  LUT='/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/LUT_NLCD2n.csv', output_raster='n_overland.tif')
    # print ("Progress -->  The NLCD was reclassified to get the Manning's n for overland")
    #
    # # 9. soil_data
    # os.chdir(simulation_folder)
    # step9_soil_files = download_soil_data_for_pytopkapi('mask.tif')
    # print ("Progress -->  The soil files were created after querrying NRCS database using R")

    # STEP1: :TODO edit run instance making procedures
    run_1 = pytopkapi_run_instance(user_name=user_name,
                                   simulation_name=simulation_name,
                                   simulation_start_date=simulation_start_date,
                                   simulation_end_date=simulation_end_date,
                                   USGS_gage=USGS_gage,
                                   timestep=timestep,
                                   threshold=threshold,

                                   mask_fname='mask.tif', simulation_folder=simulation_folder,
                                   # channel_manning_fname=channel_manning_fname,
                                   overland_manning_fname='m_overland.tif',
                                   hillslope_fname='slope.tif', dem_fname='fel.tif', channel_network_fname='src.tif',
                                   flowdir_fname='p.tif', pore_size_dist_fname='PSD.tif',
                                   bubbling_pressure_fname='BBL.tif',
                                   resid_moisture_content_fname='RSM.tif', sat_moisture_content_fname='SSM.tif',
                                   conductivity_fname='ksat.tif', soil_depth_fname='sd.tif')
    # rain_fname=rain_fname, et_fname=et_fname, runoff_fname=runoff_fname)
    print ("Progress -->  Run instance created")

    # # STEP2: Create TIFF files
    # raster_file_dict = delineate_watershed_to_get_complete_raster_set()

    raster_file_dict = {'hillslope_fname': 'slope.tif',
                        'pore_size_dist_fname': 'PSD.tif',
                        'channel_network_fname': 'src.tif',
                        'bubbling_pressure_fname': 'BBL.tif',
                        'sat_moisture_content_fname': 'SSM.tif',
                        'conductivity_fname': 'ksat.tif',
                        'dem_fname': 'fel.tif',
                        'resid_moisture_content_fname': 'RSM.tif',
                        # 'channel_manning_fname': channel_manning_fname,
                        'soil_depth_fname': 'sd.tif',
                        'flowdir_fname': 'p.tif',
                        'mask_fname': 'mask.tif',
                        'overland_manning_fname': 'n_overland.tif'}  # :TODO change this so that the function creates the dict

    # # STEP3: Create forcing files
    # forcing_files_dict = createforcingfilesforpytopkapi(mask_tiff=raster_file_dict['mask_fname'])
    forcing_files_dict = {'runoff_fname': u'http://129.123.9.159:20199/files/data/user_6/Runoff.dat',
                          'rain_fname': u'http://129.123.9.159:20199/files/data/user_6/rainfields.h5',
                          'et_fname': u'http://129.123.9.159:20199/files/data/user_6/ET.h5'}  # :TODO change this so that the function creates the dict

    # update the locations of forcing files
    # run_1.path_to_Q_observed ='http://129.123.9.159:20199/files/data/user_6/Runoff.dat'# forcing_files_dict['runoff']
    # run_1.path_to_rain_h5 = 'http://129.123.9.159:20199/files/data/user_6/rainfields.h5' #forcing_files_dict['rain']
    # run_1.path_to_et_h5 = 'http://129.123.9.159:20199/files/data/user_6/ET.h5'# forcing_files_dict['et']

    print ("Progress -->  Geospatial file received. and forcing files CHEATED")
    print ("Progress -->  Forcing files CHEATED")

    pytopkapi_input_files = run_1.prepare_supporting_ini(raster_file_dict=raster_file_dict)

    print ("Progress -->  Supporting ini files created")

    for file in pytopkapi_input_files:
        print (file)

    # # STEP4: Run the model
    outlet_ID, no_of_cell = get_outletID_noOfCell(pytopkapi_input_files['cell_param'])
    fake_rain_h5 = create_rain_ET_from_cell_no(simulation_folder, no_of_cell)

    running_instance = run_1.run_the_model(topkapi_ini=pytopkapi_input_files['TOPKAPI_ini'])
    # if running_instance['simulation_run'] == True:
    #     file_out =running_instance['results']



    # # STEP5 get just the simulated hydrograph
    ndar_Qc_out = ut.read_one_array_hdf(running_instance['results'], 'Channel', 'Qc_out')
    ar_Qsim = ndar_Qc_out[1:, outlet_ID]
    print (ar_Qsim)

    # # # STEP5: Retreive simulated and observed hydrograph
    # ar_date, ar_Qsim, error_checking_param = run_1.get_hydrograph()
    # print (ar_Qsim)

    # # STEP6 Soil Moisture Maps
    # plot_soil_moisture_maps.run(os.path.join(simulation_folder, pytopkapi_input_files['plot_soil_moisture_ini']))
    plot_soil_moisture_maps.run(pytopkapi_input_files['plot_soil_moisture_ini'])

    # # STEP7 Copy PyTOPKAPI files to user directory,so that the directory contains all files necessary for model run
    from distutils.dir_util import copy_tree
    copy_tree('/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/pytopkapi_for_users/',
              simulation_folder)

    # # LAST STEP: zip the diretory to return to user. Delete the working directory
    zip_dir(simulation_folder, output_zip)
    delete_working_uuid_directory(simulation_folder)

    return {'success': 'True'}


def runpytopkapi3(user_name, simulation_name, simulation_start_date, simulation_end_date, USGS_gage, timestep,
                  threshold,
                  # channel_manning_fname,
                  overland_manning_fname,
                  hillslope_fname, dem_fname, channel_network_fname, mask_fname, flowdir_fname,
                  pore_size_dist_fname, bubbling_pressure_fname, resid_moisture_content_fname,
                  sat_moisture_content_fname, conductivity_fname, soil_depth_fname,
                  output_txt='pytopkpai_model_files_metadata.txt'):
    """
    :param inputs as string:  Dictionary. Inputs from Tethys, or user requesting the service
    :return: Timeseries file- hydrograph, or list of input files if the user only wants input files
    """

    import pytopkapi
    from pytopkapi.results_analysis import plot_Qsim_Qobs_Rain, plot_soil_moisture_maps
    import pytopkapi.utils as ut

    simulation_folder = generate_uuid_file_path();
    print ("Progress -->  Working folder created")
    os.chdir(simulation_folder)

    # STEP1:
    run_1 = pytopkapi_run_instance(user_name=user_name,
                                   simulation_name=simulation_name,
                                   simulation_start_date=simulation_start_date,
                                   simulation_end_date=simulation_end_date,
                                   USGS_gage=USGS_gage,
                                   timestep=timestep,
                                   threshold=threshold,

                                   mask_fname=mask_fname, simulation_folder=simulation_folder,
                                   # channel_manning_fname=channel_manning_fname,
                                   overland_manning_fname=overland_manning_fname,
                                   hillslope_fname=hillslope_fname, dem_fname=dem_fname,
                                   channel_network_fname=channel_network_fname,
                                   flowdir_fname=flowdir_fname, pore_size_dist_fname=pore_size_dist_fname,
                                   bubbling_pressure_fname=bubbling_pressure_fname,
                                   resid_moisture_content_fname=resid_moisture_content_fname,
                                   sat_moisture_content_fname=sat_moisture_content_fname,
                                   conductivity_fname=conductivity_fname, soil_depth_fname=soil_depth_fname)
    # rain_fname=rain_fname, et_fname=et_fname, runoff_fname=runoff_fname)
    print ("Progress -->  Run instance created")

    # # # STEP3a: get the TIFF files, a
    raster_file_dict = run_1.raster_file_dict

    # # STEP2: Create forcing files
    abstract_pcp(startDate=simulation_start_date, endDate=simulation_end_date, cell_size=run_1.cell_size,
                 input_watershed=run_1.mask_fname, climate_variable='prcp', output_nc_fname='ppt.nc')
    str1 = 'python3.4 /home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/create_hdf5_rain.py '
    str2 = '-i ppt.nc -m '
    str3 = mask_fname
    str4 = ' -o ' + simulation_folder
    os.system(str1 + str2 + str3 + str4)
    # create_pytopkapi_hdf5_from_nc(nc_f='ppt.nc', mask_tiff=mask_fname, output_folder=simulation_folder)
    print ('Progress --> forcing files created')

    # # STEP3b get forcing files dictionary
    # # Update files
    # rain_fname = '/home/ahmet/ciwater/usu_data_service/workspace/73a3da5399944413b46b0ce7d882fa8c/rainfields.h5'
    # et_fname = '/home/ahmet/ciwater/usu_data_service/workspace/73a3da5399944413b46b0ce7d882fa8c/ET.h5'
    # runoff_fname = None
    # run_1.rain_fname =  rain_fname
    # run_1.et_fname = et_fname
    # run_1.forcing_files_dict =  {
    #                           'rain_fname':  rain_fname,
    #                           'et_fname':  et_fname  }

    pytopkapi_input_files = run_1.prepare_supporting_ini(raster_file_dict=raster_file_dict)
    print ("Progress -->  Supporting ini files created")

    # # STEP4: Run the model
    outlet_ID, no_of_cell = get_outletID_noOfCell(pytopkapi_input_files['cell_param'])

    running_instance = run_1.run_the_model(topkapi_ini=pytopkapi_input_files['TOPKAPI_ini'])
    # if running_instance['simulation_run'] == True:
    #     file_out =running_instance['results']


    # # STEP5 get just the simulated hydrograph
    ndar_Qc_out = ut.read_one_array_hdf(running_instance['results'], 'Channel', 'Qc_out')
    ar_Qsim = ndar_Qc_out[1:, outlet_ID]
    print (ar_Qsim)

    # # # STEP5: Retreive simulated and observed hydrograph
    # ar_date, ar_Qsim, error_checking_param = run_1.get_hydrograph()
    # print (ar_Qsim)

    # # STEP6 Soil Moisture Maps
    # plot_soil_moisture_maps.run(os.path.join(simulation_folder, pytopkapi_input_files['plot_soil_moisture_ini']))
    plot_soil_moisture_maps.run(pytopkapi_input_files['plot_soil_moisture_ini'])

    # :TODO A function to change the ini files, from absolute path to relative


    # # STEP7 Copy PyTOPKAPI files to user directory,so that the directory contains all files necessary for model run
    import shutil
    # from distutils.dir_util import copy_tree
    # copy_tree('/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/pytopkapi_for_users/',
    #           simulation_folder)

    # zip directory that are not file
    all_dirs = [os.path.join(simulation_folder, o) for o in os.listdir(simulation_folder) if
                os.path.isdir(os.path.join(simulation_folder, o))]
    for dir in all_dirs:
        output_zip_name = simulation_folder + '/' + os.path.basename(dir) + '.zip'
        zip_dir(dir, path_file_zip=output_zip_name)
        shutil.rmtree(dir, ignore_errors=True)

    # upload the directory to HydroShare
    hs_res_id_created = push_to_hydroshare(simulation_name=simulation_name, data_folder=simulation_folder)

    # # # LAST STEP: zip the diretory to return to user. Delete the working directory
    # print ('Trying to zip everything now')
    # zip_dir(simulation_folder, output_zip)
    with open(output_txt, 'w') as f:
        f.write(str(hs_res_id_created))

    print ('deleting working directory now')
    delete_working_uuid_directory(simulation_folder)

    return {'success': 'True'}


def runpytopkapi4(user_name, simulation_name, simulation_start_date, simulation_end_date, USGS_gage, timestep,
                  threshold,
                  # channel_manning_fname,
                  overland_manning_fname,
                  hillslope_fname, dem_fname, channel_network_fname, mask_fname, flowdir_fname,
                  pore_size_dist_fname, bubbling_pressure_fname, resid_moisture_content_fname,
                  sat_moisture_content_fname, conductivity_fname, soil_depth_fname,
                  output_hs_rs_id_txt='pytopkpai_model_files_metadata.txt', output_q_sim_txt='q_sim_cfs.txt'):
    '''

    :param inputs as string:  Dictionary. Inputs from Tethys, or user requesting the service
    :return: Timeseries file- hydrograph, or list of input files if the user only wants input files

    :param user_name:                       string
    :param simulation_name:                 string
    :param simulation_start_date:           string, format = '01/25/2010'
    :param simulation_end_date:
    :param USGS_gage:
    :param timestep:
    :param threshold:
    :param overland_manning_fname:
    :param hillslope_fname:
    :param dem_fname:
    :param channel_network_fname:
    :param mask_fname:
    :param flowdir_fname:
    :param pore_size_dist_fname:
    :param bubbling_pressure_fname:
    :param resid_moisture_content_fname:
    :param sat_moisture_content_fname:
    :param conductivity_fname:
    :param soil_depth_fname:
    :param output_hs_rs_id_txt:
    :param output_q_sim_txt:
    :return:
    '''

    import pytopkapi
    from pytopkapi.results_analysis import plot_Qsim_Qobs_Rain, plot_soil_moisture_maps
    import pytopkapi.utils as ut

    simulation_folder = os.path.split(output_hs_rs_id_txt)[0];
    print ("Progress -->  Working folder created")
    os.chdir(simulation_folder)

    # STEP1:
    run_1 = pytopkapi_run_instance(user_name=user_name,
                                   simulation_name=simulation_name,
                                   simulation_start_date=simulation_start_date,
                                   simulation_end_date=simulation_end_date,
                                   USGS_gage=USGS_gage,
                                   timestep=timestep,
                                   threshold=threshold,

                                   mask_fname=mask_fname, simulation_folder=simulation_folder,
                                   # channel_manning_fname=channel_manning_fname,
                                   overland_manning_fname=overland_manning_fname,
                                   hillslope_fname=hillslope_fname, dem_fname=dem_fname,
                                   channel_network_fname=channel_network_fname,
                                   flowdir_fname=flowdir_fname, pore_size_dist_fname=pore_size_dist_fname,
                                   bubbling_pressure_fname=bubbling_pressure_fname,
                                   resid_moisture_content_fname=resid_moisture_content_fname,
                                   sat_moisture_content_fname=sat_moisture_content_fname,
                                   conductivity_fname=conductivity_fname, soil_depth_fname=soil_depth_fname)
    # rain_fname=rain_fname, et_fname=et_fname, runoff_fname=runoff_fname)
    print ("Progress -->  Run instance created")

    # # # STEP3a: get the TIFF files, a
    raster_file_dict = run_1.raster_file_dict

    # # # STEP2: Create forcing files
    # abstract_pcp(startDate=simulation_start_date, endDate=simulation_end_date, cell_size=run_1.cell_size,
    #              input_watershed=run_1.mask_fname, climate_variable='prcp', output_nc_fname='ppt.nc')

    calculate_rain_ET_from_daymet(startDate=simulation_start_date, endDate=simulation_end_date,
                                  input_raster=run_1.mask_fname, input_dem=run_1.dem_fname, cell_size=run_1.cell_size,
                                  output_et_reference_fname='ET_reference.nc', output_rain_fname='rain.nc')

    str1 = 'python3.4 /home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/createpytopkapiforcingfile.py '
    str2 = '-ippt ppt.nc -iet ET_reference.nc -m ' + mask_fname + ' -o ' + simulation_folder + ' -t %s' % timestep
    os.system(str1 + str2)
    # create_pytopkapi_hdf5_from_nc(nc_f='ppt.nc', mask_tiff=mask_fname, output_folder=simulation_folder)
    print ('Progress --> Forcing files created')

    # # STEP3b get forcing files dictionary
    # # Update files
    # rain_fname = '/home/ahmet/ciwater/usu_data_service/workspace/73a3da5399944413b46b0ce7d882fa8c/rainfields.h5'
    # et_fname = '/home/ahmet/ciwater/usu_data_service/workspace/73a3da5399944413b46b0ce7d882fa8c/ET.h5'
    # runoff_fname = None
    # run_1.rain_fname =  rain_fname
    # run_1.et_fname = et_fname
    # run_1.forcing_files_dict =  {
    #                           'rain_fname':  rain_fname,
    #                           'et_fname':  et_fname  }

    pytopkapi_input_files = run_1.prepare_supporting_ini(raster_file_dict=raster_file_dict)
    print ("Progress -->  Supporting ini files created")

    # # STEP4: Run the model
    outlet_ID, no_of_cell = get_outletID_noOfCell(pytopkapi_input_files['cell_param'])
    running_instance = run_1.run_the_model(topkapi_ini=pytopkapi_input_files['TOPKAPI_ini'])
    # if running_instance['simulation_run'] == True:
    #     file_out =running_instance['results']
    print ("Progress -->  Simulation Successfully ran")

    # # # STEP5 get just the simulated hydrograph
    # ndar_Qc_out = ut.read_one_array_hdf(running_instance['results'], 'Channel', 'Qc_out')

    # # # STEP5: Retreive simulated and observed hydrograph
    # ar_date, ar_Qsim, error_checking_param = run_1.get_hydrograph()
    q_obs_fname = os.path.join(simulation_folder, 'q_obs_cfs.txt')

    try:
        downloadandresampleusgsdischarge(USGS_Gage=USGS_gage, begin_date=simulation_start_date,
                                         end_date=simulation_end_date, out_fname=q_obs_fname)
        q_obs_status = True
    except:
        q_obs_status = False
        print ('Progress --> The observed discharge series for the USGS gage %s not found' % USGS_gage)

    # print (ar_Qsim)
    # # # STEP5: Alternative
    q_sim_fname = os.path.join(os.path.split(mask_fname)[0],
                               output_q_sim_txt)  # output_q_sim_txt # os.path.join(simulation_folder,output_q_sim_txt )
    str1 = 'python3.4 /home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/read_result_hydrograph.py '
    str2 = '-i %s -oid %s -d %s -t %s -oq %s ' % (
    run_1.path_to_results, str(outlet_ID), simulation_start_date, str(timestep), q_sim_fname)
    # if q obs is available, pass that on
    if q_obs_status == True:
        str2 = '-i %s -oid %s -d %s -t %s -oq %s -iq %s' % (
        run_1.path_to_results, str(outlet_ID), simulation_start_date, str(timestep), q_sim_fname, q_obs_fname)
    os.system(str1 + str2)
    print ('Progress --> Hydrograph results Read')

    # # # STEP6 Soil Moisture Maps
    # # plot_soil_moisture_maps.run(os.path.join(simulation_folder, pytopkapi_input_files['plot_soil_moisture_ini']))
    # plot_soil_moisture_maps.run( pytopkapi_input_files['plot_soil_moisture_ini'])


    # :TODO A function to change the ini files, from absolute path to relative


    # # STEP7 Copy PyTOPKAPI files to user directory,so that the directory contains all files necessary for model run
    import shutil
    # from distutils.dir_util import copy_tree
    # copy_tree('/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/pytopkapi_for_users/',
    #           simulation_folder)

    # zip directory that are not file
    all_dirs = [os.path.join(simulation_folder, o) for o in os.listdir(simulation_folder) if
                os.path.isdir(os.path.join(simulation_folder, o))]
    for dir in all_dirs:
        output_zip_name = simulation_folder + '/' + os.path.basename(dir) + '.zip'
        zip_dir(dir, path_file_zip=output_zip_name)
        shutil.rmtree(dir, ignore_errors=True)

    print ('Progress --> Folders Zipped')

    change_ini_path_to_local(folders_with_inis=simulation_folder, string_to_change_in_files=simulation_folder + '/')

    # del all the unncecessary netcdf files
    onlyfiles = [f for f in os.listdir(simulation_folder) if os.path.isfile(os.path.join(simulation_folder, f))]
    nc_fullpath = [os.path.join(simulation_folder, f) for f in onlyfiles if f.endswith('.nc')]
    for file in nc_fullpath:
        os.remove(file)

    # copy mask.tif, and the two text files
    shutil.copy2(mask_fname, simulation_folder)
    output_q_sim_txt = os.path.join(os.path.split(mask_fname)[0], output_q_sim_txt)
    shutil.copy2(output_q_sim_txt, simulation_folder)

    # upload the directory to HydroShare
    print ('Progress --> Pushing files to HydroShare. This could take while...')
    hs_res_id_created = push_to_hydroshare(simulation_name=simulation_name, data_folder=simulation_folder)
    print ('Progress --> Successfully pushed files to HydroShare. Created HS_res_ID ', hs_res_id_created)

    # # # LAST STEP: zip the diretory to return to user. Delete the working directory
    # print ('Trying to zip everything now')
    # zip_dir(simulation_folder, output_zip)

    with open(output_hs_rs_id_txt, 'w') as f:
        f.write(str(hs_res_id_created))

    # copy created
    shutil.copy2(os.path.join(os.path.split(mask_fname)[0], output_hs_rs_id_txt), simulation_folder)

    # print ('deleting working directory now')
    # delete_working_uuid_directory(simulation_folder)


    return {'success': 'True'}


def runpytopkapi5(user_name, simulation_name, simulation_start_date, simulation_end_date, USGS_gage, timestep,
                  threshold,
                  overland_manning_fname, hillslope_fname, dem_fname, channel_network_fname, mask_fname, flowdir_fname,
                  pore_size_dist_fname, bubbling_pressure_fname, resid_moisture_content_fname,
                  sat_moisture_content_fname, conductivity_fname, soil_depth_fname,
                  rain_fname, et_fname, output_response_txt='pytopkpai_responseJSON.txt'):
    '''
    CHANGES to runpytopkapi4: One response file but the format is JSON. And, rain/et netCDF are to be given
    Naming for netCDFs:
    Rainfall:     variable='prcp'   unit='m/day'
    ET:           variable='ETr'    unit='m/day'
    :param inputs as string:  Dictionary. Inputs from Tethys, or user requesting the service
    :return: Timeseries file- hydrograph, or list of input files if the user only wants input files

    :param user_name:                       string
    :param simulation_name:                 string
    :param simulation_start_date:           string, format = '01/25/2010'
    :param simulation_end_date:
    :param USGS_gage:
    :param timestep:
    :param threshold:
    :param overland_manning_fname:
    :param hillslope_fname:
    :param dem_fname:
    :param channel_network_fname:
    :param mask_fname:
    :param flowdir_fname:
    :param pore_size_dist_fname:
    :param bubbling_pressure_fname:
    :param resid_moisture_content_fname:
    :param sat_moisture_content_fname:
    :param conductivity_fname:
    :param soil_depth_fname:
    :param output_response_txt:
        :param rain_fname:
        :param et_fname:
    :return:
    '''

    simulation_folder = os.path.split(output_response_txt)[0]
    os.chdir(simulation_folder)

    run_1 = pytopkapi_run_instance(user_name=user_name, simulation_name=simulation_name,
                                   simulation_start_date=simulation_start_date, simulation_end_date=simulation_end_date,
                                   USGS_gage=USGS_gage, timestep=timestep, threshold=threshold,
                                   mask_fname=mask_fname, simulation_folder=simulation_folder,
                                   # channel_manning_fname=channel_manning_fname,
                                   overland_manning_fname=overland_manning_fname,
                                   hillslope_fname=hillslope_fname, dem_fname=dem_fname,
                                   channel_network_fname=channel_network_fname,
                                   flowdir_fname=flowdir_fname, pore_size_dist_fname=pore_size_dist_fname,
                                   bubbling_pressure_fname=bubbling_pressure_fname,
                                   resid_moisture_content_fname=resid_moisture_content_fname,
                                   sat_moisture_content_fname=sat_moisture_content_fname,
                                   conductivity_fname=conductivity_fname, soil_depth_fname=soil_depth_fname,
                                   rain_fname=rain_fname, et_fname=et_fname)

    # some variables that need to be defined before we go any further
    raster_file_dict = run_1.raster_file_dict
    pytopkapi_input_files = run_1.prepare_supporting_ini(raster_file_dict=raster_file_dict)
    outlet_ID, no_of_cell = get_outletID_noOfCell(pytopkapi_input_files['cell_param'])
    q_obs_fname = os.path.join(simulation_folder, 'q_obs_cfs.txt')
    q_sim_fname = os.path.join(simulation_folder, 'q_sim_cfs.txt')

    str1 = 'python3.4 /home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/createpytopkapiforcingfile.py '
    str2 = '-ippt ' + run_1.rain_fname + ' -iet ' + run_1.et_fname + ' -m ' + mask_fname + ' -o ' + simulation_folder + ' -t %s' % timestep
    os.system(str1 + str2)

    run_detail = run_1.run_the_model(topkapi_ini=pytopkapi_input_files['TOPKAPI_ini'])

    try:
        downloadanddailyusgsdischarge(USGS_Gage=USGS_gage, begin_date=simulation_start_date,
                                         end_date=simulation_end_date, out_fname=q_obs_fname)

        q_obs_status = True

        # update q_obs in run detail json file
        with open(run_1.run_detail_JSON, 'r') as oldfile:
            old_data = json.load(oldfile)

        startDate = current_timestep_date = datetime.strptime(simulation_start_date, '%m/%d/%Y')
        discharge_list = read_hydrograph(q_obs_fname)  # something like [1,2,3,3]
        yr_mon_day_hr_min_discharge_list = []
        for value in discharge_list:
            current_timestep_date = current_timestep_date + timedelta(hours=timestep)
            yr_mon_day_hr_min_discharge_list.append(
                [current_timestep_date.year, current_timestep_date.month, current_timestep_date.day,
                 current_timestep_date.hour, current_timestep_date.minute, float(value)])
        old_data['observed_discharge'] = yr_mon_day_hr_min_discharge_list

        with open(run_1.run_detail_JSON, 'w') as updatefile:
            json.dump(old_data, updatefile)


    except Exception as E:
        print (E)
        q_obs_status = False
        print ('Progress --> The observed discharge series for the USGS gage %s not found' % USGS_gage)

    str3 = 'python3.4 /home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/read_result_hydrograph.py '
    str4 = '-i %s -oid %s -d %s -t %s -oq %s ' % (
    run_1.path_to_results, str(outlet_ID), simulation_start_date, str(timestep), q_sim_fname)
    # if q obs is available, pass that on
    if q_obs_status == True:
        str4 = '-i %s -oid %s -d %s -t %s -oq %s -iq %s' % (
        run_1.path_to_results, str(outlet_ID), simulation_start_date, str(timestep), q_sim_fname, q_obs_fname)
    os.system(str3 + str4)

    run_1.write_run_detail(run_detail=run_detail, first_time=False)

    # Change absolute path to relative
    change_ini_path_to_local(folders_with_inis=simulation_folder, string_to_change_in_files=simulation_folder + '/')

    # Del all the unncecessary netcdf files
    onlyfiles = [f for f in os.listdir(simulation_folder) if os.path.isfile(os.path.join(simulation_folder, f))]
    nc_fullpath = [os.path.join(simulation_folder, f) for f in onlyfiles if f.endswith('.nc')]
    for file in nc_fullpath:
        os.remove(file)

    # Upload the directory to HydroShare
    hs_res_id_created = push_to_hydroshare(simulation_name=simulation_name, data_folder=simulation_folder)

    # # Write q_obs and q_sim to JSON file, to be returned to the user / app
    # run_1.write_json_response(output_response_txt, hs_res_id_created, q_obs_status)


    add_an_element_to_json(run_1.run_detail_JSON, elementName='hs_res_id_created', elementValue=hs_res_id_created)
    shutil.copy(run_1.run_detail_JSON, output_response_txt)

    return {'success': 'True'}


def loadpytopkapi(hs_res_id, output_response_txt='pytopkpai_responseJSON.txt'):
    '''
    A function to load pytopkapi files from HydroShare. Specifically, just load the q_sim.txt
    :param hs_res_id:
    :param output_response_txt:
    :return:
    '''
    working_folder = os.path.split(output_response_txt)[0]
    pull_one_file_from_hydroshare(hs_id=hs_res_id, fname='run_info.txt', output_folder=working_folder)

    shutil.copy( os.path.join(working_folder,'run_info.txt') , output_response_txt)


    return {'success': 'True'}


def modifypytopkapi(fac_l, fac_ks, fac_n_o, fac_n_c, fac_th_s,
                    pvs_t0, vo_t0, qc_t0, kc,
                    hs_res_id, output_response_txt='pytopkpai_responseJSON.txt'):
    '''
    A function to load pytopkapi files from HydroShare. Specifically, just load the q_sim.txt
    :param hs_res_id:
    :param output_hs_rs_id_txt:
    :param output_q_sim_txt:
    :return:
    '''
    temp_folder = os.path.split(output_response_txt)[0]

    return_dict = pull_from_hydroshare(hs_resource_id=hs_res_id, output_folder=temp_folder)

    if return_dict['contains_pytopkapi_file'] != True:
        return {'success': 'False', 'message': 'TOPKAPI related filesnames not found'}

    # folder where the files are downloaded from HydroShare
    simulation_folder = os.path.join(temp_folder, hs_res_id, hs_res_id, 'data', 'contents')
    os.chdir(simulation_folder)

    # geth the start date, and timestep from the q_sim_cfs.txt
    ar_qsim = np.genfromtxt('q_sim_cfs.txt', delimiter=",")
    simulation_start_date = '%s/%s/%s' % (int(ar_qsim[0][1]), int(ar_qsim[0][2]), int(ar_qsim[0][0]))
    timestep = int(ar_qsim[1][2] - ar_qsim[0][2]) * 24

    # run instance
    run_modified = pytopkapi_run_instance(simulation_folder=simulation_folder, modifying=True)

    # point to forcing files
    run_modified.forcing_files_dict = {
        'rain_fname': os.path.join(simulation_folder, 'rainfields.h5'),
        'et_fname': os.path.join(simulation_folder, 'ET.h5')}

    # re-run the model
    run_detail = run_modified.run_the_model(topkapi_ini=os.path.join(simulation_folder, run_modified.TOPKAPI_ini_name),
                                            new_calib_param={'fac_l': fac_l, 'fac_ks': fac_ks, 'fac_n_o': fac_n_o,
                                                             'fac_n_c': fac_n_c, 'fac_th_s': fac_th_s},
                                            new_numeric_param={'pvs_t0': pvs_t0, 'vo_t0': vo_t0, 'qc_t0': qc_t0,
                                                               'kc': kc},
                                            cell_param_file=os.path.join(simulation_folder,run_modified.cell_param_name))

    # # STEP4: Run the model
    outlet_ID, no_of_cell = get_outletID_noOfCell(os.path.join(simulation_folder, run_modified.cell_param_name))
    mask_fname = os.path.join(simulation_folder, 'mask.tif')

    # # q_sim = os.path.join(os.path.split(mask_fname)[0] , output_q_sim_txt ) #output_q_sim_txt # os.path.join(simulation_folder,output_q_sim_txt )
    # str1 = 'python3.4 /home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/read_result_hydrograph.py '
    # str2 = '-i %s -oid %s -d %s -t %s -oq %s '% (run_modified.path_to_results, str(outlet_ID), simulation_start_date, str(timestep), output_q_sim_txt )
    # os.system(str1+ str2 )
    # print ('Progress --> Hydrograph results Read')


    q_obs_fname = os.path.join(simulation_folder, 'q_obs_cfs.txt')
    q_sim_fname = os.path.join(simulation_folder, 'q_sim_cfs.txt')

    str1 = 'python3.4 /home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/read_result_hydrograph.py '
    str2 = '-i %s -oid %s -d %s -t %s -oq %s ' % (
    run_modified.path_to_results, str(outlet_ID), simulation_start_date, str(timestep), output_response_txt)
    # if q obs is available, pass that on
    if os.path.exists(q_obs_fname):
        str2 = '-i %s -oid %s -d %s -t %s -oq %s -iq %s' % (
        run_modified.path_to_results, str(outlet_ID), simulation_start_date, str(timestep), q_sim_fname, q_obs_fname)
    os.system(str1 + str2)

    run_modified.write_run_detail(run_detail=run_detail)

    change_ini_path_to_local(folders_with_inis=simulation_folder, string_to_change_in_files=simulation_folder + '/')

    # replace changed files in HydroShare
    list_of_files_to_update = [output_response_txt, 'cell_param.dat', 'TOPKAPI.ini', 'q_sim_cfs.txt',
                               'results.h5']  #:TODO needs addition of JSON file
    replace_file_in_hydroshare(existing_hs_id=hs_res_id, data_folder=simulation_folder,
                               list_of_files_to_update=list_of_files_to_update)

    add_an_element_to_json(run_modified.run_detail_JSON, elementName='hs_res_id_created', elementValue=hs_res_id)
    shutil.copy(run_modified.run_detail_JSON, output_response_txt)

    return {'success': 'True'}


def reclassify_raster_with_LUT(input_raster, LUT, output_raster='reclassified_raster.tif', delimiter=","):
    # LUT_array = np.genfromtxt(LUT, delimiter=delimiter)
    import pandas as pd
    LUT_array = np.array(pd.read_csv(LUT))

    calc_list = ['%s*(A==%s)' % (new, old) for old, new in LUT_array]
    calc_string = "+".join(calc_list)
    cmdString = 'gdal_calc.py -A %s --outfile=%s --calc="' % (
    input_raster, output_raster) + calc_string + '" --type=Float32'
    return call_subprocess(cmdString, 'reclassify raster')


def delineate_watershed_to_get_complete_raster_set(input_DEM_raster=None, input_outlet_shapefile=None,
                                                   output_raster='mask.tif', output_outlet_shapefile=None,
                                                   stream_threshold=None,
                                                   # output_strahler_order_raster = 'strlr.tif',
                                                   output_contributing_area_raster='ad8.tif',
                                                   output_fill_raster='fel.tif', output_flow_direction_raster='p.tif',
                                                   output_slope_raster='sd8.tif', output_stream_raster='src.tif',
                                                   ):
    """TauDEM doesn't take compressed file; uncompress file
        ToDO:  Check compression first"""
    temp_raster = 'temp.tif'

    # # had to uncomment it to make it work (Prasanna)
    # retDictionary = uncompressRaster(input_DEM_raster, temp_raster)
    # if retDictionary['success']=="False":
    #     return retDictionary

    # the input stream threshold is in Area in km2, not no of cells
    # convert input stream thrsold from area in km2 to m2, then divide by cell_size area
    cell_size = get_cellSize(input_DEM_raster)
    stream_threshold = int(stream_threshold * 1000000 / (cell_size * cell_size))

    # input_raster = os.path.splitext(input_DEM_raster)[0]      #remove the .tif
    # pit remove
    cmdString = "pitremove -z " + input_DEM_raster + " -fel " + output_fill_raster
    retDictionary = call_subprocess(cmdString, 'pit remove')
    if retDictionary['success'] == "False":
        return retDictionary

    # d8 flow dir
    # TODO: slope should be in degree.
    cmdString = "d8flowdir -fel " + output_fill_raster + " -sd8 " + output_slope_raster + " -p " \
                + output_flow_direction_raster
    retDictionary = call_subprocess(cmdString, 'd8 flow direction')
    if retDictionary['success'] == "False":
        return retDictionary

    # d8 contributing area without outlet shape file
    cmdString = "aread8 -p " + output_flow_direction_raster + " -ad8 " + output_contributing_area_raster + " -nc"  # check the effect of -nc
    # -o "\ +input_outletshp
    retDictionary = call_subprocess(cmdString, 'd8 contributing area')
    if retDictionary['success'] == "False":
        return retDictionary

    # Get statistics of ad8 file to determine threshold

    # # TODO: Strahler order, from the stream already made
    # cmdString = "gridnet -p " + output_flow_direction_raster + " -gord " + output_strahler_order_raster
    # retDictionary = call_subprocess(cmdString, 'Strahler order of streams')
    # if retDictionary['success']=="False":
    #     return retDictionary


    # Stream definition by threshold
    cmdString = "threshold -ssa " + output_contributing_area_raster + " -src " + output_stream_raster + " -thresh " + str(
        stream_threshold)
    retDictionary = call_subprocess(cmdString, 'Stream definition by threshold')
    if retDictionary['success'] == "False":
        return retDictionary

    # move outlets to stream
    cmdString = "moveoutletstostrm -p " + output_flow_direction_raster + " -src " + output_stream_raster + " -o " \
                + input_outlet_shapefile + " -om " + output_outlet_shapefile
    retDictionary = call_subprocess(cmdString, 'move outlet to stream')

    if retDictionary['success'] == "False":
        return retDictionary

    # Add projection to moved outlet ---TauDEM excludes the projection from moved outlet; check
    driverName = "ESRI Shapefile"
    driver = ogr.GetDriverByName(driverName)
    dataset = driver.Open(input_outlet_shapefile)
    layer = dataset.GetLayer()
    srs = layer.GetSpatialRef()
    baseName = os.path.splitext(output_outlet_shapefile)[0]
    projFile = baseName + ".prj"
    # srsString = "+proj=utm +zone="+str(utmZone)+" +ellps=GRS80 +datum=NAD83 +units=m"
    # srs = osr.SpatialReference()
    # srs.ImportFromEPSG(epsgCode)
    # srs.ImportFromProj4(srsString)
    srs.MorphFromESRI()
    file = open(projFile, "w")
    file.write(srs.ExportToWkt())
    file.close()
    # d8 contributing area with outlet shapefile
    cmdString = "aread8 -p " + output_flow_direction_raster + " -ad8 " + output_contributing_area_raster + " -o " + output_outlet_shapefile + " -nc"
    retDictionary = call_subprocess(cmdString, 'd8 contributing area with outlet shapefile')
    if retDictionary['success'] == "False":
        return retDictionary

    # watershed grid file using the threshold function
    cmdString = "threshold -ssa " + output_contributing_area_raster + " -src " + output_raster + " -thresh 1"
    ##" python \"C:/Python34/Lib/site-packages/osgeo/gdal_calc.py\" -A "+input_raster+"ad8.tif --outfile="+output_WS_raster+" --calc=A/A"
    cmdString = 'gdal_calc.py -A ' + output_contributing_area_raster + " --outfile=" + output_raster + ' --calc=A/A'

    retDictionary = call_subprocess(cmdString, "watershed grid computation")

    # # Reclassify stream raster to get the mannings n for stream
    # reclassify_raster_with_LUT(input_raster = output_stream_raster,reclassify_overland_or_stream='stream', output_raster=output_mannings_n_stream_raster)
    return retDictionary


def download_soil_data_for_pytopkapi(Watershed_Raster, output_f_file='f.tif', output_k_file='ko.tif',
                                     output_dth1_file='dth1.tif', output_dth2_file='dth2.tif',
                                     output_psif_file='psif.tif', output_sd_file='sd.tif', output_tran_file='trans.tif',
                                     output_bubbling_pressure_file='BBL.tif',
                                     output_pore_size_distribution_file="PSD.tif",
                                     output_residual_soil_moisture_file='RSM.tif',
                                     output_saturated_soil_moisture_file='SSM.tif',
                                     output_ksat_rawls_file='ksat_rawls.tif', output_ksat_hz_file='ksat_hz.tif'
                                     ):
    head, tail = os.path.split(str(Watershed_Raster))
    Base_Data_dir_Soil = Base_Data_dir_Soil = os.path.join('/home/ahmet/hydosdata/gSSURGO', 'soil_mukey_westernUS.tif')
    Soil_script = os.path.join(
        '/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/Extract_Soil_Data_pytopkapi2.r')

    os.chdir(head)
    wateshed_Dir = str(head)
    watershed_raster_name = str(tail)
    soil_output_file = os.path.join(head, 'Soil_mukey.tif')

    # convert reference watershed raster for which the soil files are desired to shapefile
    cmd1 = "gdaltindex clipper.shp" + " " + Watershed_Raster
    os.system(cmd1)

    # clip soil_mukey.tif file with the shapefile (for watershed, created in the step earlier)
    cdf = "gdalwarp -cutline clipper.shp -dstnodata NA -crop_to_cutline" + " " + Base_Data_dir_Soil + " " + "Soil_mukey.tif"
    os.system(cdf)

    # run R script by passing in arguments so that the R script creates
    heads, tails = os.path.split(str(soil_output_file))
    cmd_str1 = "Rscript %s %s %s " % (Soil_script, wateshed_Dir, tails)
    cmd_str2 = " ".join(str(item) for item in
                        [output_f_file, output_k_file, output_dth1_file, output_dth2_file, output_psif_file,
                         output_sd_file, output_tran_file, output_bubbling_pressure_file,
                         output_pore_size_distribution_file, output_residual_soil_moisture_file,
                         output_saturated_soil_moisture_file, output_ksat_rawls_file, output_ksat_hz_file])

    os.system(cmd_str1 + cmd_str2)

    return {'success': 'True', 'message': 'download soil data successful'}


def download_soil_data_for_pytopkapi3(Watershed_Raster, output_f_file='f.tif', output_k_file='ko.tif',
                                      output_dth1_file='dth1.tif', output_dth2_file='dth2.tif',
                                      output_psif_file='psif.tif', output_sd_file='sd.tif',
                                      output_tran_file='trans.tif',
                                      output_bubbling_pressure_file='BBL.tif',
                                      output_pore_size_distribution_file="PSD.tif",
                                      output_residual_soil_moisture_file='RSM.tif',
                                      output_saturated_soil_moisture_file='SSM.tif',
                                      output_ksat_rawls_file='ksat_rawls.tif', output_ksat_hz_file='ksat_hz.tif'
                                      ):
    '''
    This will download soil file. COmpared to previous funciton, it does not give 3outputs; f.tif, to.tif, tans.tif .
    Will also use Extract_Soil_Data_pytopkapi3.r
    :param Watershed_Raster:
    :param output_f_file:
    :param output_k_file:
    :param output_dth1_file:
    :param output_dth2_file:
    :param output_psif_file:
    :param output_sd_file:
    :param output_tran_file:
    :param output_bubbling_pressure_file:
    :param output_pore_size_distribution_file:
    :param output_residual_soil_moisture_file:
    :param output_saturated_soil_moisture_file:
    :param output_ksat_rawls_file:
    :param output_ksat_hz_file:
    :return:
    '''

    head, tail = os.path.split(str(Watershed_Raster))
    Base_Data_dir_Soil = Base_Data_dir_Soil = os.path.join('/home/ahmet/hydosdata/gSSURGO', 'soil_mukey_westernUS.tif')
    Soil_script = os.path.join(
        '/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/Extract_Soil_Data_pytopkapi3.r')

    os.chdir(head)
    wateshed_Dir = str(head)
    watershed_raster_name = str(tail)
    soil_output_file = os.path.join(head, 'Soil_mukey.tif')

    # convert reference watershed raster for which the soil files are desired to shapefile
    cmd1 = "gdaltindex clipper.shp" + " " + Watershed_Raster
    os.system(cmd1)

    # clip soil_mukey.tif file with the shapefile (for watershed, created in the step earlier)
    cdf = "gdalwarp -cutline clipper.shp -dstnodata NA -crop_to_cutline" + " " + Base_Data_dir_Soil + " " + "Soil_mukey.tif"
    os.system(cdf)

    # run R script by passing in arguments so that the R script creates
    heads, tails = os.path.split(str(soil_output_file))
    cmd_str1 = "Rscript %s %s %s " % (Soil_script, wateshed_Dir, tails)
    cmd_str2 = " ".join(str(item) for item in [output_dth1_file, output_dth2_file, output_psif_file, output_sd_file,
                                               output_bubbling_pressure_file, output_pore_size_distribution_file,
                                               output_residual_soil_moisture_file, output_saturated_soil_moisture_file,
                                               output_ksat_rawls_file, output_ksat_hz_file])

    os.system(cmd_str1 + cmd_str2)

    return {'success': 'True', 'message': 'download soil data successful'}


def download_soil_data_for_pytopkapi4(Watershed_Raster, output_dth1_file='dth1.tif', output_dth2_file='dth2.tif',
                                      output_psif_file='psif.tif', output_sd_file='sd.tif',
                                      output_bubbling_pressure_file='BBL.tif',
                                      output_pore_size_distribution_file="PSD.tif",
                                      output_residual_soil_moisture_file='RSM.tif',
                                      output_saturated_soil_moisture_file='SSM.tif',
                                      output_ksat_LUT_file='ksat_LUT.tif',
                                      output_ksat_ssurgo_wtd_file='ksat_ssurgo_wtd.tif',
                                      output_ksat_ssurgo_min_file='ksat_ssurgo_min.tif',
                                      output_hydrogrp_file='hydrogrp.tif'):
    '''
    This will download soil file. COmpared to previous funciton, it does not give 3outputs; f.tif, to.tif, tans.tif .
    Will also use Extract_Soil_Data_pytopkapi3.r
    :param Watershed_Raster:
    :param output_f_file:
    :param output_k_file:
    :param output_dth1_file:
    :param output_dth2_file:
    :param output_psif_file:
    :param output_sd_file:
    :param output_tran_file:
    :param output_bubbling_pressure_file:
    :param output_pore_size_distribution_file:
    :param output_residual_soil_moisture_file:
    :param output_saturated_soil_moisture_file:
    :param output_ksat_LUT_file:
    :param output_ksat_ssurgo_wtd_file:
    :param output_ksat_ssurgo_min_file:
    :param output_hydrogrp_file:
    :return:
    '''

    head, tail = os.path.split(str(Watershed_Raster))
    Base_Data_dir_Soil = Base_Data_dir_Soil = os.path.join('/home/ahmet/hydosdata/gSSURGO', 'soil_mukey_westernUS.tif')
    Soil_script = os.path.join(
        '/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/Extract_Soil_Data_pytopkapi4.r')

    os.chdir(head)
    wateshed_Dir = str(head)
    watershed_raster_name = str(tail)
    soil_output_file = os.path.join(head, 'Soil_mukey.tif')

    # convert reference watershed raster for which the soil files are desired to shapefile
    cmd1 = "gdaltindex clipper.shp" + " " + Watershed_Raster
    os.system(cmd1)

    # clip soil_mukey.tif file with the shapefile (for watershed, created in the step earlier)
    cdf = "gdalwarp -cutline clipper.shp -dstnodata NA -crop_to_cutline" + " " + Base_Data_dir_Soil + " " + "Soil_mukey.tif"
    os.system(cdf)

    # run R script by passing in arguments so that the R script creates
    heads, tails = os.path.split(str(soil_output_file))
    cmd_str1 = "Rscript %s %s %s " % (Soil_script, wateshed_Dir, tails)
    cmd_str2 = " ".join(str(item) for item in [output_dth1_file, output_dth2_file, output_psif_file, output_sd_file,
                                               output_bubbling_pressure_file, output_pore_size_distribution_file,
                                               output_residual_soil_moisture_file, output_saturated_soil_moisture_file,
                                               output_ksat_LUT_file, output_ksat_ssurgo_wtd_file,
                                               output_ksat_ssurgo_min_file,
                                               output_hydrogrp_file])

    os.system(cmd_str1 + cmd_str2)

    return {'success': 'True', 'message': 'download soil data successful'}


def download_soil_data_for_pytopkapi5(Watershed_Raster, output_dth1_file='dth1.tif', output_dth2_file='dth2.tif',
                                      output_psif_file='psif.tif', output_sd_file='sd.tif',
                                      output_bubbling_pressure_file='BBL.tif',
                                      output_pore_size_distribution_file="PSD.tif",
                                      output_residual_soil_moisture_file='RSM.tif',
                                      output_saturated_soil_moisture_file='SSM.tif',
                                      output_ksat_LUT_file='ksat_LUT.tif',
                                      output_ksat_ssurgo_wtd_file='ksat_ssurgo_wtd.tif',
                                      output_ksat_ssurgo_min_file='ksat_ssurgo_min.tif',
                                      output_hydrogrp_file='hydrogrp.tif'):
    '''
    This will download soil file. COmpared to previous funciton, it does not give 3outputs; f.tif, to.tif, tans.tif .
    Will also use Extract_Soil_Data_pytopkapi3.r
    :param Watershed_Raster:
    :param output_f_file:
    :param output_k_file:
    :param output_dth1_file:
    :param output_dth2_file:
    :param output_psif_file:
    :param output_sd_file:
    :param output_tran_file:
    :param output_bubbling_pressure_file:
    :param output_pore_size_distribution_file:
    :param output_residual_soil_moisture_file:
    :param output_saturated_soil_moisture_file:
    :param output_ksat_LUT_file:
    :param output_ksat_ssurgo_wtd_file:
    :param output_ksat_ssurgo_min_file:
    :param output_hydrogrp_file:
    :return:
    '''

    head, tail = os.path.split(str(Watershed_Raster))
    Base_Data_dir_Soil = Base_Data_dir_Soil = os.path.join('/home/ahmet/hydosdata/gSSURGO', 'soil_mukey_westernUS.tif')
    Soil_script = os.path.join(
        '/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/Extract_Soil_Data_pytopkapi5.r')

    os.chdir(head)
    wateshed_Dir = str(head)
    watershed_raster_name = str(tail)
    soil_output_file = os.path.join(head, 'Soil_mukey.tif')

    in_raster = get_raster_detail(input_raster=Watershed_Raster)
    xmin, xmax, ymin, ymax, ncol, nrow = in_raster['minx'], in_raster['maxx'], in_raster['miny'], in_raster['maxy'], in_raster['ncol'], in_raster['nrow']

    # run R script by passing in arguments so that the R script creates
    heads, tails = os.path.split(str(soil_output_file))
    cmd_str1 = "Rscript %s %s %s " % (Soil_script, wateshed_Dir, tails)
    cmd_str2 = " ".join(str(item) for item in [output_dth1_file, output_dth2_file, output_psif_file, output_sd_file,
                                               output_bubbling_pressure_file, output_pore_size_distribution_file,
                                               output_residual_soil_moisture_file, output_saturated_soil_moisture_file,
                                               output_ksat_LUT_file, output_ksat_ssurgo_wtd_file,
                                               output_ksat_ssurgo_min_file, output_hydrogrp_file,
                                               xmin, xmax, ymin, ymax, ncol, nrow, Watershed_Raster
                                               ])

    os.system(cmd_str1 + cmd_str2)

    return {'success': 'True', 'message': 'download soil data successful'}



def command_string_to_zip(command_string=None, in_zip=None, out_zip=None):
    # idea : let user access HydroDS terminal. He can use zip file to input, and will get zip output
    if in_zip != None:
        import zipfile
        with zipfile.ZipFile(in_zip, "r") as zip_ref:
            zip_ref.extractall()
    return call_subprocess(command_string, command_string + ' could not be executed')













