import numpy as np
import os, json
from osgeo import gdal
import createpytopkapiforcingfile

def get_raster_detail(tif_file):
    dx= None
    ncol =None
    nrow = None
    bands= None
    try:

        from osgeo import gdal
        ds = gdal.Open(tif_file)

        x0, dx, fy, y0, fx, dy = ds.GetGeoTransform()
        ncol= ds.RasterXSize
        nrow = ds.RasterYSize
        bands= ds.RasterCount

        print ('Progress --> Cell size calculated is %s m' % dx)
    except:
        dx = ""
        print ("Either no GDAL, or no tiff file")
    return {'cell_size':dx, 'x':dx, 'ncol':ncol, 'nrow':nrow, 'bands':bands}


def change_date_from_mmddyyyy_to_yyyyddmm(in_date):
    '''
    :param in_date:         accepts date of formate '01/25/2010'
    :return:                converts the date to formate: '2010-01-25'
    '''
    from datetime import datetime
    in_date_element = datetime.strptime(in_date , '%m/%d/%Y')
    out_date = "%s-%s-%s"%(in_date_element.year, in_date_element.month, in_date_element.day)
    return out_date

def downloadandresampleusgsdischarge(USGS_Gage, begin_date='10/01/2010', end_date='12/30/2010',out_fname='q_obs_cfs.txt',
                                         output_unit='cfs',resampling_time = '1D', resampling_method='mean'):
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

    import urllib
    import pandas as pd

    print ('Input begin date',begin_date)
    begin_date = change_date_from_mmddyyyy_to_yyyyddmm(in_date=begin_date)
    end_date = change_date_from_mmddyyyy_to_yyyyddmm(in_date=end_date)
    print ('Edited begin date', begin_date)
    print ('Required format is yyyy-mm-dd')
    urlString3 = 'http://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&format=rdb&site_no=%s&period=&begin_date=%s&end_date=%s'%(USGS_Gage, begin_date, end_date)

    response = urllib.request.urlopen(urlString3)  # instance of the file from the URL
    html = response.read()                  # reads the texts into the variable html

    print ('Progress --> HTML read for the observed timeseries')
    with open('Q_raw.txt', 'wb') as f:
        f.write(html)

    df = pd.read_csv('Q_raw.txt', delimiter='\t' , skiprows=28, names=['agency_cd', 'USGS_Station_no', 'datatime', 'timezone', 'Q_cfs','Quality'])

    # convert datetime from string to datetime
    df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2],errors='ignore')

    # create a different dataframe with just the values and datetime
    df_datetime_val = df[['datatime', 'Q_cfs']]

    # convert the values to series
    values = []
    dates = []

    # add values to the list a
    multiplier = 1.0
    for v in df_datetime_val.iloc[:,1]:
        if output_unit.lower()=='cumecs' or 'cumec':
            multiplier =  0.028316846592
        values.append(float(v)* multiplier)

    # add datatime to list b
    for v in df_datetime_val.iloc[:, 0]:
        dates.append(v)

    # prepare a panda series
    ts = pd.Series(values, index=dates)

    # resample to daily or whatever
    # ts_mean = ts.resample('1D', how='mean') #or
    # ts_mean = ts.resample('1D').mean()
    ts_mean = ts.resample(resampling_time, how=resampling_method)


    # save
    ts_mean.to_csv(out_fname)
    print ('Progress --> Output creatad for observed file at %s'%out_fname)
    return {'success':True, 'output_file':out_fname}


def read_hydrograph(input_q='/home/ahmet/ciwater/usu_data_service/workspace/1a525d929c0e4624b03c083d198f2d38/844254f63dbd4ef1ac7c56ff97a4db96/844254f63dbd4ef1ac7c56ff97a4db96/data/contents/q_sim_cfs.txt'):
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

    print ('Just the hydrogrpah list is', q)
    return q



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

    print (' minx, maxx, miny, maxy', minx, maxx, miny, maxy)
    return {'success':'True', 'minx':minx, 'miny':miny,'maxx':maxx,'maxy':maxy}


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
    Soil_script = os.path.join(
        '/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/Extract_Soil_Data_pytopkapi5.r')

    os.chdir(head)
    wateshed_Dir = str(head)
    watershed_raster_name = str(tail)

    soil_output_file = os.path.join(head, 'Soil_mukey.tif')

    bbox = get_box_from_tif(Watershed_Raster, output_json='json.txt')
    minx, miny, maxx, maxy = bbox['minx'],bbox['miny'],bbox['maxx'],bbox['maxy']

    raster_info = get_raster_detail(Watershed_Raster)

    # run R script by passing in arguments so that the R script creates
    heads, tails = os.path.split(str(soil_output_file))
    cmd_str1 = "Rscript %s %s %s " % (Soil_script, wateshed_Dir, tails)
    cmd_str2 = " ".join(str(item) for item in [output_dth1_file, output_dth2_file, output_psif_file, output_sd_file,
                                               output_bubbling_pressure_file, output_pore_size_distribution_file,
                                               output_residual_soil_moisture_file, output_saturated_soil_moisture_file,
                                               output_ksat_LUT_file, output_ksat_ssurgo_wtd_file,
                                               output_ksat_ssurgo_min_file,
                                               output_hydrogrp_file, minx, maxx, miny, maxy, raster_info['ncol'],raster_info['nrow']  ])

    os.system(cmd_str1 + cmd_str2)
    print ('Successful')

    return {'success': 'True', 'message': 'download soil data successful'}

#get_box_from_tif(input_raster='plunge.tif', output_json='a.json')

ppt = createpytopkapiforcingfile.create_pytopkapi_hdf5_from_nc_unit_mm_per_timestep(nc_rain='/home/ahmet/ciwater/static/media/data/user_6/SWIT.nc', #SWIT_SQ9tRwQ
                                                                                    nc_et='/home/ahmet/ciwater/static/media/data/user_6/SWIT.nc',
                                                                                    mask_tiff='/home/ahmet/ciwater/static/media/data/user_6/mask.tif',
                                                                                    timestep_in_hr=24, output_folder="", source='ueb')