# cutting plane method for maximization IP, simplex from question 1 used from file simplex.py

import numpy as np
import copy
import sys
import math
from simplex import *

INPUT_PATH = "./Q3/"
OUTPUT_PATH = "./Q3/"
ip_file = INPUT_PATH +  "input.txt"
op_file = OUTPUT_PATH + "output.txt"
    
def print_cp_results(message, opt_val, opt_vect, num_x, num_cuts):
    if message != "Optimal":
        print(message)
        print(num_cuts)
        return
    print(-round(opt_val, 6))
    x = []
    ind = 1
    for key in opt_vect:
        x.append(int(opt_vect[key]))
        if ind >= num_x:
            break
        ind+=1
    print(*x)
    # print(num_cuts)

if __name__ == "__main__":
    if len(sys.argv)>1:
        ip_file = INPUT_PATH + sys.argv[1]
        op_file = OUTPUT_PATH + sys.argv[1]
    # get initial values
    ini_A, num_row_A, num_col_A, ini_b, ini_c, ini_num_slack, ini_num_artificial, ini_B, ini_vars = initialize(ip_file)
    # print(A, b, c)
    # print(num_row_A, num_col_A)
    # print(num_slack, num_artificial)
    # print(B, vars)
    flag = 1
    num_gomory = 0
    ini_c = [-i for i in ini_c]

    message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix = lp_solve(ini_A, ini_b, ini_c, ini_B, ini_vars, ini_num_artificial, ini_num_slack)
    # print(message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix)
    # print()
    # print(message, opt_val_vector, basic_variables, A_matrix, x_B, c_j)
    while flag:
        # check if decision vars are fractional
        if message == "Optimal":
            frac_var = ""
            for key in opt_val_vector:
                if opt_val_vector[key] != int(opt_val_vector[key]) and key.find("x") != -1:
                    frac_var = key
                    break
            if frac_var == "":
                flag = 0
                sys.stdout = open(op_file, 'w')
                print_cp_results(message, opt_val, opt_val_vector, num_col_A, num_gomory)
                sys.stdout.close()
                break
        else:
            sys.stdout = open(op_file, 'w')
            print_cp_results(message, opt_val, opt_val_vector, num_col_A, num_gomory)
            sys.stdout.close()
            flag = 0

        # updating A, b, c, B, vars, num_artificial, num_slack according to gomory cut
        # restart the LP by after adding new constraint of gomory cut
        num_gomory+=1
        ini_num_slack+=1
        bs = copy.deepcopy(basic_variables)
        xb = copy.deepcopy(x_B)
        a = copy.deepcopy(A_matrix)

        vars = ini_vars
        vars.append("Sg"+str(num_gomory))

        basic_variables = ini_B
        basic_variables.append("Sg"+str(num_gomory))

        frac_eq_ind = bs.index(frac_var)

        x_B = ini_b
        x_B.append(math.floor(xb[frac_eq_ind]))

        c_j.append(0.0)

        num_c = 0
        A_matrix = ini_A
        for i in range(0, len(A_matrix)):
            A_matrix[i].append(0.0)
            num_c = len(A_matrix[i])
        new_constr = [0.0] * num_c

        ind = 0
        for val in a[frac_eq_ind]:
            new_constr[ind] = math.floor(val)
            ind += 1
        new_constr[-1] = 1
        A_matrix.append(new_constr)

        # print(num_gomory)
        # print("new constr: ", new_constr)
        message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix = lp_solve(A_matrix, x_B, ini_c, basic_variables, vars, ini_num_artificial, ini_num_slack)
        # print(message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix)
        # print()



