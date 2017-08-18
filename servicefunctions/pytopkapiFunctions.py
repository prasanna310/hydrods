import numpy, os
import argparse

def hello_world(a=None,b=None, test_output="test_helloworld.txt"):
    hello_dict = {'success': 'True', 'message': str(a+b)}
    ar = numpy.array([0,0,3,1,0])
    numpy.savetxt(test_output, ar)
    print ('Numpy text file successfully created')
    print ('This is where the files are saved by default (i think) ',os.getcwd())
    return hello_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--first_input', required=True, help='Just input some float')
    parser.add_argument('-b', '--second_input', required=True, help='Just input some float')
    parser.add_argument('-o', '--test_output', required=True, help='Name of output file that will be created in server')
    args = parser.parse_args()

    hello_world(args.first_input, args.second_input, args.test_output)