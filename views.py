import logging
import json
import requests

from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError as DRF_ValidationError
from rest_framework.exceptions import NotAuthenticated

from usu_data_service.servicefunctions.terrainFunctions import *
from usu_data_service.servicefunctions.watershedFunctions import *
from usu_data_service.servicefunctions.netcdfFunctions import *
from usu_data_service.servicefunctions.canopyFunctions import *
from usu_data_service.servicefunctions.static_data import *
from usu_data_service.topnet_data_service.TOPNET_Function import CommonLib
from usu_data_service.serializers import *
from usu_data_service.models import *
from usu_data_service.utils import *
from usu_data_service.local_settings import *
from usu_data_service.capabilities import *

from usu_data_service.servicefunctions import testFileHydroDS
from usu_data_service.pytopkapi_data_service import servicefunctions_pytopkapi

WESTERN_US_DEM = os.path.join(STATIC_DATA_ROOT_PATH, 'subsetsource/nedWesternUS.tif')

logger = logging.getLogger(__name__)

funcs = {

    'combinerasters':
        {
            'function_to_execute': combineRasters,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'combined.tif'}],
            'user_inputs': [],
            'user_file_inputs': ['input_raster1', 'input_raster2'],
            'validator': CombineRastersRequestValidator
        },
    'computerasteraspect':
        {
            'function_to_execute': computeRasterAspect,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'aspect.tif'}],
            'user_inputs': [],
            'user_file_inputs': ['input_raster'],
            'validator': ComputeRasterAspectRequestValidator
        },
    'computerasterslope':
        {
            'function_to_execute': computeRasterSlope,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'slope.tif'}],
            'user_inputs': [],
            'user_file_inputs': ['input_raster'],
            'validator': ComputeRasterSlopeRequestValidator
        },
    'concatenatenetcdf':
        {
            'function_to_execute': concatenate_netCDF,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'concatenated.nc'}],
            'user_inputs': [],
            'user_file_inputs': ['input_netcdf1', 'input_netcdf2'],
            'validator': ConcatenateNetCDFRequestValidator
        },
    'convertnetcdfunits':
        {
            'function_to_execute': convert_netcdf_units,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'converted_units.nc'}],
            'user_inputs': ['variable_name', 'variable_new_units', 'multiplier_factor', 'offset'],
            'user_file_inputs': ['input_netcdf'],
            'validator': ConvertNetCDFUnitsRequestValidator
        },
    'createoutletshapefile':
        {
            'function_to_execute': create_OutletShape_Wrapper,
            'file_inputs': [],
            'file_outputs': [{'output_shape_file_name': 'outlet.shp'}],
            'user_inputs': ['outletPointX', 'outletPointY'],
            'user_file_inputs': [],
            'validator': CreateOutletShapeRequestValidator
        },
    'delineatewatershedatshape':
        {
            'function_to_execute': delineate_Watershed_atShapeFile,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'watershed.tif'}, {'output_outlet_shapefile': 'moveout.shp'}],
            'user_inputs': ['stream_threshold'],
            'user_file_inputs': ['input_DEM_raster', 'input_outlet_shapefile'],
            'validator': DelineateWatershedAtShapeFileRequestValidator
        },
    'delineatewatershedatxy':
        {
            'function_to_execute': delineate_Watershed_TauDEM,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'watershed.tif'}, {'output_outlet_shapefile': 'moveout.shp'}],
            'user_inputs': ['epsg_code', 'stream_threshold', 'outlet_point_x', 'outlet_point_y'],
            'user_file_inputs': ['input_DEM_raster'],
            'validator': DelineateWatershedAtXYRequestValidator
        },
    'downloadstreamflow':
        {
            'function_to_execute': CommonLib.download_streamflow,
            'file_inputs': [],
            'file_outputs': [{'output_streamflow': 'streamflow_calibration.dat'}],
            'user_inputs': ['USGS_gage', 'Start_Year', 'End_Year'],
            'user_file_inputs': [],
            'validator': DownloadStreamflowRequestValidator
        },
    'getcanopyvariable':
        {
            'function_to_execute': get_canopy_variable,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'canopy.nc'}],
            'user_inputs': ['variable_name'],
            'user_file_inputs': ['in_NLCDraster'],
            'validator': GetCanopyVariableRequestValidator
        },
    # cannot confirm registration
    'getcanopyvariables':
        {
            'function_to_execute': get_canopy_variables,
            'file_inputs': [],
            'file_outputs': [{'out_ccNetCDF': 'canopy_cc.nc'}, {'out_hcanNetCDF': 'canopy_hcan.nc'},
                             {'out_laiNetCDF': 'canopy_lai.nc'}],
            'user_inputs': [],
            'user_file_inputs': ['in_NLCDraster'],
            'validator': GetCanopyVariablesRequestValidator
        },
    'netcdfrenamevariable':
        {
            'function_to_execute': netCDF_rename_variable,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'rename_varname.nc'}],
            'user_inputs': ['input_varname', 'output_varname'],
            'user_file_inputs': ['input_netcdf'],
            'validator': NetCDFRenameVariableRequestValidator
        },
    'projectandcliprastertoreference':
        {
            'function_to_execute': project_and_clip_raster,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'project_clip.tif'}],
            'user_inputs': [],
            'user_file_inputs': ['input_raster', 'reference_raster'],
            'validator': ProjectClipRasterRequestValidator
        },
    'projectnetcdf':
        {
            'function_to_execute': project_netCDF_UTM_NAD83,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'projected.nc'}],
            'user_inputs': ['utm_zone', 'variable_name'],
            'user_file_inputs': ['input_netcdf'],
            'validator': ProjectNetCDFRequestValidator
        },
    'projectraster':
        {
            'function_to_execute': project_raster_UTM_NAD83,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'projected.tif'}],
            'user_inputs': ['utmZone'],
            'user_file_inputs': ['input_raster'],
            'validator': ProjectRasterRequestValidator
        },
    'projectresamplerasterepsg':
        {
            'function_to_execute': project_and_resample_Raster_EPSG,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'projected_resampled.tif'}],
            'user_inputs': ['epsg_code', 'dx', 'dy', 'resample'],
            'user_file_inputs': ['input_raster'],
            'validator': ProjectResampleRasterEPSGRequestValidator
        },
    'projectresamplerasterutm':
        {
            'function_to_execute': project_and_resample_Raster_UTM_NAD83,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'projected_resampled.tif'}],
            'user_inputs': ['utm_zone', 'dx', 'dy', 'resample'],
            'user_file_inputs': ['input_raster'],
            'validator': ProjectResampleRasterUTMRequestValidator
        },
    'projectshapefileepsg':
        {
            'function_to_execute': project_shapefile_EPSG,
            'file_inputs': [],
            'file_outputs': [{'output_shape_file': 'shape_proj.shp'}],
            'user_inputs': ['epsg_code'],
            'user_file_inputs': ['input_shape_file'],
            'validator': ProjectShapeFileEPSGRequestValidator
        },
    'projectshapefileutm':
        {
            'function_to_execute': project_shapefile_UTM_NAD83,
            'file_inputs': [],
            'file_outputs': [{'output_shape_file': 'shape_proj.shp'}],
            'user_inputs': ['utm_zone'],
            'user_file_inputs': ['input_shape_file'],
            'validator': ProjectShapeFileUTMRequestValidator
        },
    'projectsubsetresamplenetcdftoreferencenetcdf':
        {
            'function_to_execute': project_subset_and_resample_netcdf_to_reference_netcdf,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'proj_subset_resample.nc'}],
            'user_inputs': ['variable_name'],
            'user_file_inputs': ['input_netcdf', 'reference_netcdf'],
            'validator': ProjectSubsetResampleNetCDFRequestValidator
        },
    'rastersubset':
        {
            'function_to_execute': get_raster_subset,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'subset.tif'}],
            'user_file_inputs': ['input_raster'],
            'user_inputs': ['xmin', 'ymax', 'xmax', 'ymin'],
            'validator': rastersubsetRequestValidator
        },
    'rastertonetcdf':
        {
            'function_to_execute': rasterToNetCDF,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'output.nc'}],
            'user_inputs': [],
            'user_file_inputs': ['input_raster'],
            'validator': RasterToNetCDFRequestValidator
        },
    'rastertonetcdfrenamevariable':
        {
            'function_to_execute': rasterToNetCDF_rename_variable,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'output.nc'}],
            'user_inputs': ['increasing_x', 'increasing_y', 'output_varname'],
            'user_file_inputs': ['input_raster'],
            'validator': RasterToNetCDFVariableRequestValidator
        },
    'resamplenetcdftoreferencenetcdf':
        {
            'function_to_execute': resample_netcdf_to_reference_netcdf,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'resample.nc'}],
            'user_inputs': ['variable_name'],
            'user_file_inputs': ['input_netcdf', 'reference_netcdf'],
            'validator': ResampleNetCDFRequestValidator
        },
    'resampleraster':
        {
            'function_to_execute': resample_Raster,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'resample.tif'}],
            'user_inputs': ['dx', 'dy', 'resample'],
            'user_file_inputs': ['input_raster'],
            'validator': ResampleRasterRequestValidator
        },
    'reversenetcdfyaxis':
        {
            'function_to_execute': reverse_netCDF_yaxis,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'reverse_yaxis.nc'}],
            'user_inputs': [],
            'user_file_inputs': ['input_netcdf'],
            'validator': ReverseNetCDFYaxisRequestValidator
        },
    'reversenetcdfyaxisandrenamevariable':
        {
            'function_to_execute': reverse_netCDF_yaxis_and_rename_variable,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'reverse_yaxis.nc'}],
            'user_inputs': ['input_varname', 'output_varname'],
            'user_file_inputs': ['input_netcdf'],
            'validator': ReverseNetCDFYaxisAndRenameVariableRequestValidator
        },
    'subsetnetcdfbytime':
        {
            'function_to_execute': get_netCDF_subset_TimeDim,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'subset_time_based.nc'}],
            'user_inputs': ['time_dim_name', 'start_time_index', 'end_time_index'],
            'user_file_inputs': ['input_netcdf'],
            'validator': SubsetNetCDFByTimeDimensionRequestValidator
        },
    'subsetnetcdftoreference':
        {
            'function_to_execute': subset_netCDF_to_reference_raster,
            'file_inputs': [],
            'file_outputs': [{'output_netcdf': 'subset.nc'}],
            'user_inputs': [],
            'user_file_inputs': ['input_netcdf', 'reference_raster'],
            'validator': SubsetNetCDFToReferenceRequestValidator
        },
    'subsetprism':
        {
            'function_to_execute': CommonLib.subsetprism,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'subset.tif'}],
            'user_file_inputs': [],
            'user_inputs': ['xmin', 'ymax', 'xmax', 'ymin'],
            'validator': subsetprismRequestValidator
        },
    'subsetprojectresamplerasterepsg':
        {
            'function_to_execute': subset_project_and_resample_Raster_EPSG,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'subset_proj_resample.tif'}],
            'user_file_inputs': ['input_raster'],
            'user_inputs': ['xmin', 'ymax', 'xmax', 'ymin', 'dx', 'dy', 'resample', 'epsg_code'],
            'validator': SubsetProjectResampleRasterEPSGRequestValidator
        },
    'subsetprojectresamplerasterutm':
        {
            'function_to_execute': subset_project_and_resample_Raster_UTM_NAD83,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'subset_proj_resample.tif'}],
            'user_file_inputs': ['input_raster'],
            'user_inputs': ['xmin', 'ymax', 'xmax', 'ymin', 'dx', 'dy', 'resample'],
            'validator': SubsetProjectResampleRasterRequestValidator
        },
    'subsetrastertobbox':
        {
            'function_to_execute': get_raster_subset,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'subset.tif'}],
            'user_file_inputs': ['input_raster'],
            'user_inputs': ['xmin', 'ymax', 'xmax', 'ymin'],
            'validator': SubsetDEMRequestValidator
        },
    'subsetrastertoreference':
        {
            'function_to_execute': subset_raster_to_referenceRaster,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'subset_ref.tif'}],
            'user_inputs': [],
            'user_file_inputs': ['input_raster', 'reference_raster'],
            'validator': SubsetRasterToReferenceRequestValidator
        },
    # cannot confirm registrations
    'subsetUSGSNEDDEM':
        {
            'function_to_execute': subset_USGS_NED_DEM,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'subset_usgs_ned_dem.tif'}],
            'user_file_inputs': [],
            'user_inputs': ['xmin', 'ymax', 'xmax', 'ymin'],
            'validator': SubsetUSGSNEDDEMRequestValidator
        },

    'taudemwatersheddelineation':
        {
            'function_to_execute': CommonLib.watershed_delineation,
            'file_inputs': [],
            'file_outputs': [],
            'user_inputs': ['a', 'b'],
            'user_file_inputs': [],
            'validator': helloworldvalidator
        },


    # TOPNET FUNCTIONS
    'createbasinparameter':
        {
            'function_to_execute': CommonLib.BASIN_PARAM,
            'file_inputs': [],
            'file_outputs': [{'output_basinfile': 'basinpars.txt'}],
            'user_inputs': [],
            'user_file_inputs': ['DEM_Raster', 'f_raster', 'k_raster', 'dth1_raster', 'dth2_raster', 'sd_raster',
                                 'psif_raster', 'tran_raster', 'lulc_raster', 'lutlc', 'lutkc', 'Watershed_Raster',
                                 'parameter_specficationfile', 'nodelinksfile'],
            'validator': createbasinparameterdataRequestValidator
        },
    'createlatlonfromxy':
        {
            'function_to_execute': CommonLib.create_latlonfromxy,
            'file_inputs': [],
            'file_outputs': [{'output_latlonfromxyfile': 'latlongfromxy.txt'}],
            'user_inputs': [],
            'user_file_inputs': ['Watershed_Raster'],
            'validator': createlatlonfromxydataRequestValidator
        },
    'createparmfile':
        {
            'function_to_execute': CommonLib.Create_Parspcfile,
            'file_inputs': [],
            'file_outputs': [{'output_parspcfile': 'parspc.txt'}],
            'user_inputs': [],
            'user_file_inputs': ['Watershed_Raster'],
            'validator': createparmfiledataRequestValidator
        },
    'createrainweight':
        {
            'function_to_execute': CommonLib.Create_rain_weight,
            'file_inputs': [],
            'file_outputs': [{'output_rainweightfile': 'rainweights.txt'}],
            'user_inputs': [],
            'user_file_inputs': ['Watershed_Raster', 'Rain_gauge_shapefile', 'annual_rainfile', 'nodelink_file'],
            'validator': createrainweightdataRequestValidator
        },
    'dist_wetness_distribution':
        {
            'function_to_execute': CommonLib.DISTANCE_DISTRIBUTION,
            'file_inputs': [],
            'file_outputs': [{'output_distributionfile': 'distribution.txt'}],
            'user_inputs': [],
            'user_file_inputs': ['Watershed_Raster', 'SaR_Raster', 'Dist_Raster'],
            'validator': dist_wetness_distributiondataRequestValidator
        },
    'downloadclimatedata':
        {
            'function_to_execute': CommonLib.daymet_download,
            'file_inputs': [],
            'file_outputs': [{'output_rainfile': 'rain.dat', 'output_temperaturefile': 'tmaxtmintdew.dat',
                              'output_cliparfile': 'clipar.dat', 'output_gagefile': 'rain_gage.shp'}],
            'user_inputs': ['Start_Year', 'End_Year'],
            'user_file_inputs': ['Watershed_Raster'],
            'validator': DownloadClimatedataRequestValidator
        },
    'downloadsoildata':
        {
            'function_to_execute': CommonLib.download_Soil_Data,
            'file_inputs': [],
            'file_outputs': [{'output_f_file': 'f.tif', 'output_k_file': 'ko.tif', 'output_dth1_file': 'dth1.tif',
                              'output_dth2_file': 'dth2.tif', 'output_psif_file': 'psif.tif',
                              'output_sd_file': 'sd.tif', 'output_tran_file': 'trans.tif'}],
            'user_inputs': [],
            'user_file_inputs': ['Watershed_Raster'],
            'validator': DownloadSoildataRequestValidator
        },
    'getlanduselandcoverdata':
        {
            'function_to_execute': CommonLib.getLULCdata,
            'file_inputs': [],
            'file_outputs': [{'output_LULCRaster': 'LULC_Watershed.tif'}],
            'user_inputs': [],
            'user_file_inputs': ['Watershed_Raster'],
            'validator': getlanduselandcoverdataRequestValidator
        },
    'getprismrainfall':
        {
            'function_to_execute': CommonLib.getprismdata,
            'file_inputs': [],
            'file_outputs': [{'output_raster': 'annrain.tif'}],
            'user_inputs': [],
            'user_file_inputs': ['Watershed_Raster'],
            'validator': getprismrainfalldataRequestValidator
        },
    'reachlink':
        {
            'function_to_execute': CommonLib.REACH_LINK,
            'file_inputs': [],
            'file_outputs': [{'output_reachfile': 'rchlink.txt', 'output_nodefile': 'nodelinks.txt',
                              'output_reachareafile': 'rchareas.txt', 'output_rchpropertiesfile': 'rchproperties.txt'}],
            'user_inputs': [],
            'user_file_inputs': ['DEM_Raster', 'Watershed_Raster', 'treefile', 'coordfile'],
            'validator': ReachLinkdataRequestValidator
        },

    'watersheddelineation':
        {
            'function_to_execute': CommonLib.watershed_delineation,
            'file_inputs': [],
            'file_outputs': [
                {'output_pointoutletshapefile': 'moved_outlets.shp', 'output_watershedfile': 'Delineated_Watershed.tif',
                 'output_treefile': 'Stream_tree.txt', 'output_coordfile': 'Stream_coord.txt',
                 'output_streamnetfile': 'Streamnet.shp', 'output_slopareafile': 'SlopeAreaRatio.tif',
                 'output_distancefile': 'DistanceStream.tif'}],
            'user_inputs': ['Src_threshold', 'Min_threshold', 'Max_threshold', 'Number_threshold'],
            'user_file_inputs': ['DEM_Raster', 'Outlet_shapefile'],
            'validator': WatershedDelineationdataRequestValidator
        },





    # pytopkapi functions
    'helloworld':
        {
            'function_to_execute':testFileHydroDS.hello_world ,
            'file_inputs': [],
            'file_outputs': [{'test_output_1': 'test_helloworld_1.txt'}, {'test_output_2': 'test_helloworld_2.txt'}],
            'user_inputs': ['a', 'b'],
            'user_file_inputs': [],
            'validator': helloworldvalidator
        },

    'delineatewatershedtogetcompleterasterset':
        {
            'function_to_execute':  servicefunctions_pytopkapi.delineate_watershed_to_get_complete_raster_set,
            'file_inputs': [],
            'file_outputs': [
                # {'output_strahler_order_raster': 'strlr.tif'},
                {'output_contributing_area_raster': 'ad8.tif'},
                {'output_fill_raster': 'fel.tif'},
                {'output_flow_direction_raster': 'p.tif'},
                {'output_outlet_shapefile': 'moveout.shp'},
                {'output_raster': 'watershed.tif'},
                {'output_slope_raster': 'sd8.tif'},
                {'output_stream_raster': 'src.tif'},
                # {'output_mannings_n_stream_raster': 'n_stream.tif'}
            ],

            'user_inputs': ['stream_threshold'],
            'user_file_inputs': ['input_DEM_raster', 'input_outlet_shapefile'],
            'validator': delineatewatershedtogetcompleterastersetValidator
        },

    'downloadsoildataforpytopkapi':
        {
            'function_to_execute': CommonLib.download_soil_data_for_pytopkapi, #:todo implement all topkapi function in one place
            'file_inputs': [],
            'file_outputs': [{'output_f_file': 'f.tif', 'output_k_file': 'ko.tif', 'output_dth1_file': 'dth1.tif',
                              'output_dth2_file': 'dth2.tif', 'output_psif_file': 'psif.tif',
                              'output_sd_file': 'sd.tif', 'output_tran_file': 'trans.tif',

                              'output_bubbling_pressure_file':'BBL.tif', 'output_pore_size_distribution_file':'PSD.tif',
                             'output_residual_soil_moisture_file':'RSD.tif', 'output_saturated_soil_moisture_file':'SSM.tif',
                             'output_ksat_rawls_file':'ksat_rawls.tif'
        }],
            'user_inputs': [],
            'user_file_inputs': ['Watershed_Raster'],
            'validator': downloadsoildataforpytopkapiRequestValidator
        },

    'downloadsoildataforpytopkapi2':
        {
            'function_to_execute': servicefunctions_pytopkapi.download_soil_data_for_pytopkapi,
            'file_inputs': [],
            'file_outputs': [{'output_f_file': 'f.tif', 'output_k_file': 'ko.tif', 'output_dth1_file': 'dth1.tif',
                              'output_dth2_file': 'dth2.tif', 'output_psif_file': 'psif.tif',
                              'output_sd_file': 'sd.tif', 'output_tran_file': 'trans.tif',

                              'output_bubbling_pressure_file': 'BBL.tif',
                              'output_pore_size_distribution_file': 'PSD.tif',
                              'output_residual_soil_moisture_file': 'RSD.tif',
                              'output_saturated_soil_moisture_file': 'SSM.tif',
                              'output_ksat_rawls_file': 'ksat_rawls.tif',
                              'output_ksat_hz_file': 'ksat_hz.tif'
                              }],
            'user_inputs': [],
            'user_file_inputs': ['Watershed_Raster'],
            'validator': downloadsoildataforpytopkapi2RequestValidator
        },

    'downloadsoildataforpytopkapi3':
        {
            'function_to_execute': servicefunctions_pytopkapi.download_soil_data_for_pytopkapi3,
            'file_inputs': [],
            'file_outputs': [{ 'output_dth1_file': 'dth1.tif','output_dth2_file': 'dth2.tif',
                               'output_psif_file': 'psif.tif',
                              'output_sd_file': 'sd.tif',

                              'output_bubbling_pressure_file': 'BBL.tif',
                              'output_pore_size_distribution_file': 'PSD.tif',
                              'output_residual_soil_moisture_file': 'RSD.tif',
                              'output_saturated_soil_moisture_file': 'SSM.tif',
                              'output_ksat_rawls_file': 'ksat_rawls.tif',
                              'output_ksat_hz_file': 'ksat_hz.tif'
                              }],
            'user_inputs': [],
            'user_file_inputs': ['Watershed_Raster'],
            'validator': downloadsoildataforpytopkapi3RequestValidator
        },
    'downloadsoildataforpytopkapi4':
        {
            'function_to_execute': servicefunctions_pytopkapi.download_soil_data_for_pytopkapi4,
            'file_inputs': [],
            'file_outputs': [{'output_dth1_file': 'dth1.tif',
                              'output_dth2_file': 'dth2.tif',
                              'output_psif_file': 'psif.tif',
                              'output_sd_file': 'sd.tif',

                              'output_bubbling_pressure_file': 'BBL.tif',
                              'output_pore_size_distribution_file': 'PSD.tif',
                              'output_residual_soil_moisture_file': 'RSD.tif',
                              'output_saturated_soil_moisture_file': 'SSM.tif',
                                'output_ksat_LUT_file':'ksat_LUT.tif',
                                'output_ksat_ssurgo_wtd_file':'ksat_ssurgo_wtd.tif',
                                'output_ksat_ssurgo_min_file':'ksat_ssurgo_min.tif',
                                'output_hydrogrp_file':'hydrogrp.tif',
                              }],
            'user_inputs': [],
            'user_file_inputs': ['Watershed_Raster'],
            'validator': downloadsoildataforpytopkapi4RequestValidator
        },

    'downloadsoildataforpytopkapi5':
        {
            'function_to_execute': servicefunctions_pytopkapi.download_soil_data_for_pytopkapi5,
            'file_inputs': [],
            'file_outputs': [{'output_dth1_file': 'dth1.tif',
                              'output_dth2_file': 'dth2.tif',
                              'output_psif_file': 'psif.tif',
                              'output_sd_file': 'sd.tif',

                              'output_bubbling_pressure_file': 'BBL.tif',
                              'output_pore_size_distribution_file': 'PSD.tif',
                              'output_residual_soil_moisture_file': 'RSD.tif',
                              'output_saturated_soil_moisture_file': 'SSM.tif',
                              'output_ksat_LUT_file': 'ksat_LUT.tif',
                              'output_ksat_ssurgo_wtd_file': 'ksat_ssurgo_wtd.tif',
                              'output_ksat_ssurgo_min_file': 'ksat_ssurgo_min.tif',
                              'output_hydrogrp_file': 'hydrogrp.tif',
                              }],
            'user_inputs': [],
            'user_file_inputs': ['Watershed_Raster'],
            'validator': downloadsoildataforpytopkapi5RequestValidator
        },


    'reclassifyrasterwithlut':
        {
            'function_to_execute': servicefunctions_pytopkapi.reclassify_raster_with_LUT,
            'file_inputs': [],
            'file_outputs': [  {'output_raster': 'reclassified_raster.tif'}],
            'user_inputs': ['delimiter'],
            'user_file_inputs': ['input_raster', 'LUT' ],
            'validator':ReclassifyRasterWithLUTdataRequestValidator
        },

    'runpytopkapi':
        {
            'function_to_execute': servicefunctions_pytopkapi.runpytopkapi,
            'file_inputs': [],
            'file_outputs': [{'output_zip': 'pytopkapi_model_files.zip'}],
            'user_inputs': [
                'user_name', 'simulation_name', 'simulation_start_date',
                'simulation_end_date', 'USGS_gage', 'timestep','threshold',

            ],
            'user_file_inputs': [  'mask_fname','overland_manning_fname', 'hillslope_fname' , 'dem_fname', 'channel_network_fname',
                'flowdir_fname', 'pore_size_dist_fname' , 'bubbling_pressure_fname' , 'resid_moisture_content_fname' ,
                'sat_moisture_content_fname', 'conductivity_fname', 'soil_depth_fname'
                                   # 'rain_fname','et_fname','runoff_fname',
                                   ],
            'validator': runpytopkapivalidator
        },

    'runpytopkapi2':
        {
            'function_to_execute': servicefunctions_pytopkapi.runpytopkapi2,
            'file_inputs': [],
            'file_outputs': [{'output_zip': 'pytopkapi_model_files.zip'}],
            'user_inputs': [
                'user_name', 'simulation_name', 'simulation_start_date',
                'simulation_end_date', 'USGS_gage',
                "outlet_y", "outlet_x", "box_topY", "box_bottomY", "box_rightX", "box_leftX", "timeseries_source",
                "cell_size", "model_engine",
                'timestep', 'threshold',

            ],
            'user_file_inputs': [],
            'validator': runpytopkapi2validator
        },

    'runpytopkapi3':
        {
            'function_to_execute': servicefunctions_pytopkapi.runpytopkapi3,
            'file_inputs': [],
            'file_outputs': [{'output_txt': 'pytopkpai_model_files_metadata.txt'}],
            'user_inputs': [
                'user_name', 'simulation_name', 'simulation_start_date',
                'simulation_end_date', 'USGS_gage', 'timestep', 'threshold',



            ],
            'user_file_inputs': ['mask_fname', 'overland_manning_fname', 'hillslope_fname', 'dem_fname',
                                 'channel_network_fname',
                                 'flowdir_fname', 'pore_size_dist_fname', 'bubbling_pressure_fname',
                                 'resid_moisture_content_fname',
                                 'sat_moisture_content_fname', 'conductivity_fname', 'soil_depth_fname',
                                 # 'rain_fname','et_fname','runoff_fname'
                                 ],
            'validator': runpytopkapi3validator
        },

    'runpytopkapi4':
        {
            'function_to_execute': servicefunctions_pytopkapi.runpytopkapi4,
            'file_inputs': [],
            'file_outputs': [{'output_hs_rs_id_txt': 'pytopkpai_model_files_metadata.txt',
                              'output_q_sim_txt':'output_q_sim_txt.txt'
                              }],
            'user_inputs': [
                'user_name', 'simulation_name', 'simulation_start_date',
                'simulation_end_date', 'USGS_gage', 'timestep', 'threshold',

            ],
            'user_file_inputs': ['mask_fname', 'overland_manning_fname', 'hillslope_fname', 'dem_fname',
                                 'channel_network_fname',
                                 'flowdir_fname', 'pore_size_dist_fname', 'bubbling_pressure_fname',
                                 'resid_moisture_content_fname',
                                 'sat_moisture_content_fname', 'conductivity_fname', 'soil_depth_fname',
                                 # 'rain_fname','et_fname','runoff_fname'
                                 ],
            'validator': runpytopkapi4validator
        },
    'runpytopkapi5':
        {
            'function_to_execute': servicefunctions_pytopkapi.runpytopkapi5,
            'file_inputs': [],
            'file_outputs': [{'output_response_txt': 'pytopkpai_responseJSON.txt',
                              }],
            'user_inputs': [
                'user_name', 'simulation_name', 'simulation_start_date',
                'simulation_end_date', 'USGS_gage', 'timestep', 'threshold',

            ],
            'user_file_inputs': ['mask_fname', 'overland_manning_fname', 'hillslope_fname', 'dem_fname',
                                 'channel_network_fname',
                                 'flowdir_fname', 'pore_size_dist_fname', 'bubbling_pressure_fname',
                                 'resid_moisture_content_fname',
                                 'sat_moisture_content_fname', 'conductivity_fname', 'soil_depth_fname',
                                 'rain_fname','et_fname',
                                 ],
            'validator': runpytopkapi5validator
        },

    'loadpytopkapi':
        {
            'function_to_execute': servicefunctions_pytopkapi.loadpytopkapi,
            'file_inputs': [],
            'file_outputs': [{'output_response_txt': 'pytopkpai_responseJSON.txt',
                              }],
            'user_inputs': ['hs_res_id'],
            'user_file_inputs': [],
            'validator': loadpytopkapivalidator
        },

    'modifypytopkapi':
        {
            'function_to_execute': servicefunctions_pytopkapi.modifypytopkapi,
            'file_inputs': [],
            'file_outputs': [{'output_response_txt': 'pytopkpai_responseJSON.txt',
                              }],
            'user_inputs': ['hs_res_id', 'fac_l', 'fac_ks', 'fac_n_o', 'fac_n_c','fac_th_s',
                'pvs_t0' ,'vo_t0' ,'qc_t0' ,'kc' ],
            'user_file_inputs': [],
            'validator': modifypytopkapivalidator
        },

    'abstractclimatedata':
        {
            'function_to_execute': servicefunctions_pytopkapi.abstract_climate,
            'file_inputs': [],
            'file_outputs': [{'output_vp_fname': 'output_vp.nc',
                            'output_tmin_fname': 'output_tmin.nc',
                            'output_tmax_fname': 'output_tmax.nc',
                            'output_srad_fname': 'output_srad.nc',
                            'output_prcp_fname': 'output_prcp.nc',
                            # 'output_dayl_fname': 'output_dayl.nc',
                              }],
            'user_inputs': ['startDate', 'endDate', 'cell_size'],
            'user_file_inputs': ['input_raster'],
            'validator': abstractclimatedataRequestValidator
        },

    'calculatereferenceetfromdaymet':
        {
            'function_to_execute': servicefunctions_pytopkapi.calculate_reference_et_from_netCDFs,
            'file_inputs': [],
            'file_outputs': [{'out_et_nc': 'out_et.nc',
                              }],
            'user_inputs': ['windspeed'],
            'user_file_inputs': ['dem_nc', 'srad_nc', 'tmax_nc', 'tmin_nc', 'vp_nc'],
            'validator': calculatereferenceetfromdaymetRequestValidator
        },
    'calculaterainETfromdaymet':
        {
            'function_to_execute': servicefunctions_pytopkapi.calculate_rain_ET_from_daymet,
            'file_inputs': [],
            'file_outputs': [{'output_et_reference_fname': 'ET_reference.nc',
                              'output_rain_fname':'rain.nc',
                              }],
            'user_inputs': ['startDate', 'endDate', 'cell_size'],
            'user_file_inputs': ['input_raster', 'input_dem'],
            'validator': calculaterainETfromdaymetRequestValidator
        },

    'changetimestepofforcingnetcdf':
        {
            'function_to_execute': servicefunctions_pytopkapi.change_timestep_of_forcing_netcdf,
            'file_inputs': [],
            'file_outputs': [{'output_et_reference_fname': 'ET_reference.nc',
                              'output_rain_fname': 'rain.nc',
                              }],
            'user_inputs': ['time_interval_in_hr'],
            'user_file_inputs': ['input_netcdf'],
            'validator': changetimestepofforcingnetcdfRequestValidator
        },

    'bboxfromtiff':
        {
            'function_to_execute': servicefunctions_pytopkapi.get_box_from_tif,
            'file_inputs': [],
            'file_outputs': [{'output_json': 'json_response.txt'
                              }],
            'user_inputs': [],
            'user_file_inputs': ['input_raster'],
            'validator': bboxfromtiffRequestValidator
        },
    'bboxfromshp':
        {
            'function_to_execute': servicefunctions_pytopkapi.get_box_xyxy_from_shp,
            'file_inputs': [],
            'file_outputs': [{'output_json': 'json_response.txt'
                              }],
            'user_inputs': [],
            'user_file_inputs': ['input_shp'],
            'validator': bboxfromshpRequestValidator
        },
    'outletxyfromshp':
        {
            'function_to_execute': servicefunctions_pytopkapi.get_outlet_xy_from_shp,
            'file_inputs': [],
            'file_outputs': [{'output_json': 'json_response.txt'
                              }],
            'user_inputs': [],
            'user_file_inputs': ['input_shp'],
            'validator': outletxyfromshpRequestValidator
        },
         }


class RunService(APIView):
    """
    Executes the specified service/function

    URL: /api/dataservice/{func}
    HTTP method: GET

    :param func: name of the function to execute

    The function specific parameter values needs to be passed as part of the query string

    :raises
    ValidationError: json response format: {'parameter_1': [parameter_1_error], 'parameter_2': [parameter_2_error], ..}
    """
    allowed_methods = ('GET',)

    def get(self, request, func, format=None):

        if not request.user.is_authenticated():
            raise NotAuthenticated()

        logger.info('Executing python data service function:' + func)
        params = funcs.get(func, None)

        if not params:
            return Response({'success': False, 'error': 'No such function {function_name} is '
                                                        'supported.'.format(function_name=func)})

        validator = params['validator']

        request_validator = validator(data=self.request.query_params)
        if not request_validator.is_valid():
            raise DRF_ValidationError(detail=request_validator.errors)

        subprocparams = {}
        for param_dict_item in params['file_inputs']:
            for param_name in param_dict_item:
                subprocparams[param_name] = param_dict_item[param_name]

        # generate uuid file name for each parameter in file_outputs dict
        uuid_file_path = generate_uuid_file_path()
        logger.debug('temporary uuid working directory for function ({function_name}):{w_dir}'.format(
                     function_name=func, w_dir=uuid_file_path))

        output_files = {}
        for param_dict_item in params['file_outputs']:
            for param_name in param_dict_item:
                output_file_name = request_validator.validated_data.get(param_name, param_dict_item[param_name])
                subprocparams[param_name] = os.path.join(uuid_file_path, output_file_name)
                output_files[param_name] = subprocparams[param_name]

        for p in params['user_inputs']:
            subprocparams[p] = request_validator.validated_data[p]

        # user input file can come as a url file path or just a file name
        # comes in url format for files that are stored for the user in django, copy the file to uuid temp folder
        # and then pass the uuid file path to the executing function
        # comes as a file name for static data file on the server, get the static data file path from the file name
        # and pass that file path to the executing function
        for p in params['user_file_inputs']:
            input_file = request_validator.validated_data[p]
            if is_input_file_url_path(input_file):
                uuid_input_file_path = copy_input_file_to_uuid_working_directory(uuid_file_path,
                                                                                 request_validator.validated_data[p])
                if uuid_input_file_path.endswith('.zip'):
                    unzip_shape_file(uuid_input_file_path)
                    uuid_input_file_path = uuid_input_file_path.replace('zip', 'shp')

                subprocparams[p] = uuid_input_file_path
                logger.debug('input_uuid_file_path_from_url_path:' + uuid_input_file_path)
            else:
                static_data_file_path = get_static_data_file_path(input_file)
                subprocparams[p] = static_data_file_path
                logger.debug('input_static_file_path:' + static_data_file_path)

        # execute the function
        result = params['function_to_execute'](**subprocparams)
        logger.debug('result from function ({function_name}):{result}'.format(function_name=func, result=result))

        # process function output results
        data = []
        if result['success'] == 'True':
            user = request.user if request.user.is_authenticated() else None
            data = _save_output_files_in_django(output_files, user=user)
            response_data = {'success': True, 'data': data, 'error': []}
        else:
            response_data = {'success': False, 'data': data, 'error': result['message']}

        delete_working_uuid_directory(uuid_file_path)

        return Response(data=response_data)


@api_view(['GET'])
def show_capabilities(request):
    data = get_capabilites()
    response_data = {'success': True, 'data': data, 'error': []}
    return Response(data=response_data)


@api_view(['GET'])
def show_service_info(request, func):
    data = get_service_info(service_name=func)
    if data is None:
        raise DRF_ValidationError("%s is not a supported service name" % func)
    response_data = {'success': True, 'data': data, 'error': []}
    return Response(data=response_data)

@api_view(['GET'])
def show_static_data_info(request):
    data = get_static_data_files_info()
    response_data = {'success': True, 'data': data, 'error': []}
    return Response(data=response_data)

@api_view(['POST'])
def upload_file(request):
    if not request.user.is_authenticated():
        raise NotAuthenticated()

    number_of_files = len(request.FILES)

    if number_of_files == 0:
        error_msg = {'file': 'No file was found to upload.'}
        raise DRF_ValidationError(detail=error_msg)
    elif number_of_files > 1:
        error_msg = {'file': 'More than one file was found. Only one file can be uploaded at a time.'}
        raise DRF_ValidationError(detail=error_msg)

    posted_file = request.FILES['file']
    user_file = UserFile(file=posted_file)
    user_file.user = request.user
    user_file.save()

    file_url = current_site_url() + user_file.file.url.replace('/static/media/', '/files/')
    response_data = {'success': True, 'data': file_url, 'error': []}
    return Response(data=response_data)

@api_view(['GET'])
def get_hydrogate_result_file(request):
    if not request.user.is_authenticated():
        raise NotAuthenticated()

    request_validator = GetHydrogateResultFileRequestValidator(data=request.query_params)
    if not request_validator.is_valid():
            raise DRF_ValidationError(detail=request_validator.errors)

    hydrogate_result_file_name = request_validator.validated_data['result_file_name']
    save_as_file_name = request_validator.validated_data['save_as_file_name']
    hg_download_file_url_path = 'http://129.123.41.158:20198/{file_name}'.format(file_name=hydrogate_result_file_name)
    uuid_file_path = generate_uuid_file_path()
    save_as = os.path.join(uuid_file_path, save_as_file_name)
    with open(save_as, 'wb') as file_obj:
        response = requests.get(hg_download_file_url_path, stream=True)
        if not response.ok:
            # Something went wrong
            error_msg = 'Hydrogate error. ' + response.reason + " " + response.content
            response_data = {'success': False, 'data': [], 'error': [error_msg]}
            return Response(data=response_data)

        for block in response.iter_content(1024):
            if not block:
                break
            file_obj.write(block)

    delete_user_file(request.user, save_as_file_name)
    user_file = UserFile(file=File(open(save_as, 'rb')), user=request.user)
    user_file.save()
    file_url = current_site_url() + user_file.file.url.replace('/static/media/', '/files/')
    response_data = {'success': True, 'data': file_url, 'error': []}
    logger.debug('django file url for the hydrogate result file:' + user_file.file.url)
    delete_working_uuid_directory(uuid_file_path)
    return Response(data=response_data)


@api_view(['GET'])
def show_my_files(request):
    if not request.user.is_authenticated():
        raise NotAuthenticated()

    user_files = []
    for user_file in UserFile.objects.filter(user=request.user).all():
        if user_file.file:
            user_file_url = current_site_url() + user_file.file.url.replace('/static/media/', '/files/')
            user_files.append(user_file_url)

    response_data = {'success': True, 'data': user_files, 'error': []}

    return Response(data=response_data)


@api_view(['DELETE'])
def delete_my_file(request, filename):
    if not request.user.is_authenticated():
        raise NotAuthenticated()

    for user_file in UserFile.objects.filter(user=request.user).all():
        if user_file.file.name.split('/')[2] == filename:
            user_file.file.delete()
            user_file.delete()
            logger.debug("{file_name} file deleted by user:{user_id}".format(file_name=filename,
                                                                             user_id=request.user.id))
            break

    else:
        raise NotFound()

    response_data = {'success': True, 'data': filename, 'error': []}
    return Response(data=response_data)

@api_view(['GET'])
def create_hydroshare_resource(request):
    if not request.user.is_authenticated():
        raise NotAuthenticated()

    request_validator = HydroShareCreateResourceRequestValidator(data=request.query_params)
    if not request_validator.is_valid():
        raise DRF_ValidationError(detail=request_validator.errors)

    hs_username = request_validator.validated_data['hs_username']
    hs_password = request_validator.validated_data['hs_password']
    hydroshare_auth = (hs_username, hs_password)
    file_name = request_validator.validated_data['file_name']
    resource_type = request_validator.validated_data['resource_type']
    title = request_validator.validated_data.get('title', None)
    abstract = request_validator.validated_data.get('abstract', None)
    keywords = request_validator.validated_data.get('keywords', None)
    metadata = request_validator.validated_data.get('metadata', None)

    for user_file in UserFile.objects.filter(user=request.user).all():
        if user_file.file.name.split('/')[2] == file_name:
            break
    else:
        raise NotFound()

    hs_url = 'https://www.hydroshare.org/hsapi/resource'
    payload = {'resource_type': resource_type}

    if title:
        payload['title'] = title

    if abstract:
        payload['abstract'] = abstract

    if keywords:
        for (i, kw) in enumerate(keywords):
                key = "keywords[{index}]".format(index=i)
                payload[key] = kw

    if metadata:
        payload['metadata'] = metadata

    user_folder = 'user_%s' % request.user.id
    source_file_path = os.path.join(settings.MEDIA_ROOT, 'data', user_folder, file_name)
    files = {'file': open(source_file_path, 'rb')}

    # create a resource in HydroShare
    response = requests.post(hs_url+'/?format=json', data=payload, files=files, auth=hydroshare_auth)

    if response.ok:
        response_content_dict = json.loads(response.content.decode('utf-8'))
        response_data = {'success': True, 'data': response_content_dict, 'error': []}
    else:
        err_msg = "Failed to create a resource in HydroShare.{reason}".format(reason=response.reason)
        response_data = {'success': False, 'data': [], 'error': err_msg}

    return Response(data=response_data)


@api_view(['GET'])
def zip_my_files(request):
    if not request.user.is_authenticated():
        raise NotAuthenticated()

    request_validator = ZipMyFilesRequestValidator(data=request.query_params)
    if not request_validator.is_valid():
        raise DRF_ValidationError(detail=request_validator.errors)

    files_to_zip = request_validator.validated_data['file_names']
    zip_file_name = request_validator.validated_data['zip_file_name']
    zip_file_url = zip_user_files(user=request.user, file_name_list=files_to_zip, zip_file_name=zip_file_name)

    response_data = {'success': True, 'data': {'zip_file_name': zip_file_url}, 'error': []}
    return Response(data=response_data)


def _save_output_files_in_django(output_files, user=None):
    output_files_in_django = {}

    # first delete if any of these output files already exist for the user
    for key, value in output_files.items():
        if user:
            file_to_delete = os.path.basename(value)
            # check if the output file is a shape file then we need to delete a matching zip file
            # as shapefiles are saved as zip files
            if file_to_delete.endswith('.shp'):
                file_to_delete = file_to_delete[:-4] + '.zip'
            delete_user_file(user, file_to_delete)

    for key, value in output_files.items():
        # check if it is a shape file
        ext = os.path.splitext(value)[1]
        if ext == '.shp':
            logger.debug('creating zip for shape files')
            value = create_shape_zip_file(value)

        user_file = UserFile(file=File(open(value, 'rb')))
        if user:
            user_file.user = user

        user_file.save()
        output_files_in_django[key] = current_site_url() + user_file.file.url.replace('/static/media/', '/files/')
        logger.debug('django file url for the output file:' + user_file.file.url)

    return output_files_in_django



