'''
The purpose of this script is to convert FDR tiff's into connectivity rasters
'''

import argparse, os, sys
import h5py, numpy
from datetime import datetime, timedelta


def read_hydrograph_from_results(results=None,outlet_id=None, simulation_start_date=None,timestep=24,
                                 output_qsim=None, input_q_obs=None, ):
    '''
    The output file contains array of the format:
            YYYY  MM  DD  hh  mm  q_simulated q_observed
            Both q_simulated and q_observed are in cfs
    :param results:
    :param outlet_id:
    :param simulation_start_date:
    :param timestep:
    :param output_qsim:
    :param input_q_obs:
    :return:
    '''


    # # default results location
    # if results == None:
    #     results = os.path.join(output_folder, "results.h5")
    #
    # # output path
    # if output_qsim == None:
    #     output_qsim = os.path.join(output_folder, "Q_sim.txt")

    f =  h5py.File(results, 'r')
    ndar_Qc_out = f['Channel/Qc_out'][:]
    ar_Qsim = ndar_Qc_out[:,int(outlet_id)]
    ar_Qsim[numpy.isnan(ar_Qsim)] = 0  # line added
    f.close()


    # create an array with first line as
    # 2011  01  05  0   0   15

    s = datetime.strptime(simulation_start_date, "%m/%d/%Y")
    timestep = timedelta(hours=int(timestep))
    final_array = []
    for i in range(len(ar_Qsim)-1):  # for some reason, simulated values are one more than the observed.. :TODO, fix this
        one_timestep = [s.year, s.month, s.day, s.hour, s.minute, ar_Qsim[i]/ 0.028316846592 ] # with the multiplier, output is now in cfs
        final_array.append(one_timestep)
        s = s + timestep

    # replace nans with 0
    # final_array[numpy.isnan(final_array)] = 0
    # final_array = numpy.nan_to_num(final_array)
    # print (final_array)

    # numpy.savetxt(output_qsim, X=final_array, fmt='%2d %2d %2d %2d %2d %7.3f',delimiter=',')  # was 5.1f
    numpy.savetxt(output_qsim, X=final_array, fmt='%2d,%2d,%2d,%2d,%2d,%7.3f', delimiter=',')  # was 5.1f

    print ('Progress --> Hydrograph results Read')
    return


def read_hydrograph_from_results_working_but_if_observeddatapresent_createsonemorecolumn(results=None,outlet_id=None, simulation_start_date=None,timestep=24,
                                 output_qsim=None, input_q_obs=None, ):
    '''
    The output file contains array of the format:
            YYYY  MM  DD  hh  mm  q_simulated q_observed
            Both q_simulated and q_observed are in cfs
    :param results:
    :param outlet_id:
    :param simulation_start_date:
    :param timestep:
    :param output_qsim:
    :param input_q_obs:
    :return:
    '''


    # # default results location
    # if results == None:
    #     results = os.path.join(output_folder, "results.h5")
    #
    # # output path
    # if output_qsim == None:
    #     output_qsim = os.path.join(output_folder, "Q_sim.txt")

    f =  h5py.File(results, 'r')
    ndar_Qc_out = f['Channel/Qc_out'][:]
    ar_Qsim = ndar_Qc_out[:,int(outlet_id)]
    ar_Qsim[numpy.isnan(ar_Qsim)] = 0  # line added
    f.close()

    # if q_obs is given
    if input_q_obs != None:
        f = file(input_q_obs, "r")
        str_to_save = f.read().replace('-', ",")
        f.close()

        # save it again
        f = file(input_q_obs, "w")
        f.write(str_to_save)
        f.close()

        # f = np.loadtxt(input_q_obs, delimiter=",")
        f = numpy.genfromtxt(input_q_obs, delimiter=',')
        q_obs = f[:, -1]

    # create an array with first line as
    # 2011  01  05  0   0   15

    s = datetime.strptime(simulation_start_date, "%m/%d/%Y")
    timestep = timedelta(hours=int(timestep))
    final_array = []
    for i in range(len(ar_Qsim)-1):  # for some reason, simulated values are one more than the observed.. :TODO, fix this
        one_timestep = [s.year, s.month, s.day, s.hour, s.minute, ar_Qsim[i]/ 0.028316846592 ] # with the multiplier, output is now in cfs
        if input_q_obs != None:
            one_timestep = [s.year, s.month, s.day, s.hour, s.minute, ar_Qsim[i] / 0.028316846592, q_obs[i]]
        final_array.append(one_timestep)
        s = s + timestep

    # replace nans with 0
    # final_array[numpy.isnan(final_array)] = 0
    # final_array = numpy.nan_to_num(final_array)
    # print (final_array)
    if input_q_obs != None:
        numpy.savetxt(output_qsim, X=final_array, fmt='%2d %2d %2d %2d %2d %7.3f %7.3f', delimiter=',')  # was 5.1f
    else:
        numpy.savetxt(output_qsim, X=final_array, fmt='%2d %2d %2d %2d %2d %7.3f',delimiter=',')  # was 5.1f
    print ('Progress --> Hydrograph results Read')
    return

def read_hydrograph_from_results_backup(results=None,outlet_id=None, simulation_start_date=None,timestep=24,
                                 output_qsim=None, input_q_obs=None):
    import h5py, numpy
    from datetime import  datetime, timedelta

    # # default results location
    # if results == None:
    #     results = os.path.join(output_folder, "results.h5")
    #
    # # output path
    # if output_qsim == None:
    #     output_qsim = os.path.join(output_folder, "Q_sim.txt")

    f =  h5py.File(results, 'r')
    ndar_Qc_out = f['Channel/Qc_out'][:]
    ar_Qsim = ndar_Qc_out[:,int(outlet_id)]
    ar_Qsim[numpy.isnan(ar_Qsim)] = 0  # line added
    f.close()

    # if q_obs is given
    if input_q_obs != None:
        #input_q_obs
        pass #numpy.genfromtxt()

    # create an array with first line as
    # 2011  01  05  0   0   15

    s = datetime.strptime(simulation_start_date, "%m/%d/%Y")
    timestep = timedelta(hours=int(timestep))
    final_array = []
    for i in range(len(ar_Qsim)):
        one_timestep = [s.year, s.month, s.day, s.hour, s.minute, ar_Qsim[i]/ 0.028316846592 ] # with the multiplier, output is now in cfs
        final_array.append(one_timestep)
        s = s + timestep

    # replace nans with 0
    # final_array[numpy.isnan(final_array)] = 0
    # final_array = numpy.nan_to_num(final_array)
    # print (final_array)
    numpy.savetxt(output_qsim, X=final_array, fmt='%2d %2d %2d %2d %2d %7.3f',delimiter='\t')  # was 5.1f

        # create empty file



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_results', required=True, help="input results.h5 file")
    parser.add_argument('-oid', '--outlet_id', required=True, help="Outlet ID of the cell whose discharge is desired")
    parser.add_argument('-d', '--start_date', required=True, help="Simulation start date")
    parser.add_argument('-t', '--timestep', required=False, help="Timestep for the simulation carried out")
    parser.add_argument('-iq', '--q_obs', required=False, help="Path to input q observed file (txt)")
    parser.add_argument('-oq', '--q_sim', required=False, help="Path to output q simulated file (txt) will be saved")

    args = parser.parse_args()

    read_hydrograph_from_results(results=args.input_results,
                                 outlet_id=args.outlet_id,
                                  simulation_start_date=args.start_date,
                                  timestep=args.timestep,
                                 output_qsim=args.q_sim,
                                input_q_obs = args.q_obs
                                  )





