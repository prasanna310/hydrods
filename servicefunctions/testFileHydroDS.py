from .utils import *
import os, numpy

def hello_world(a=None,b=None, test_output_1="test_helloworld_1.txt", test_output_2="test_helloworld_2.txt"):
    #print (a, b , test_output)
    # cmdString = "python3.4 pytopkapiFunctions.py -a "+ str(a) + "-b " + str(b) + "-o " + test_output
    result = a + b
    # return {'success': 'True', 'data': result}
    # return str(float(a)+(float(b)))

    ar = numpy.array([0,0,3,1,0])
    numpy.savetxt(test_output_1, ar)
    numpy.savetxt(test_output_2, ar)
    return {'success': 'True'}
    # return call_subprocess(cmdString, "Hello world")
    # return os.system(cmdString)


