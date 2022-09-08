# Branch and Bound for Minimization IP, simplex from question 1 used from file simplex.py

import numpy as np
import copy
import sys
import math
from simplex import *

INPUT_PATH = "./Q2/"
OUTPUT_PATH = "./Q2/"
ip_file = INPUT_PATH +  "input.txt"
op_file = OUTPUT_PATH + "output.txt"

def powerset(s):
    x = len(s)
    res = []
    for i in range(1 << x):
        res.append([s[j] for j in range(x) if (i & (1 << j))])
    return res

# taking A as input, converting to cut set formulation of TSP and then to canonical form for two-phased simplex
def initialize_bb(ip_file):
    inp = open(ip_file, "r")
    inp.readline()
    line = inp.readline()
    initial_basis = []
    variables = []
    # reading matrix
    num_col_A = 0
    num_row_A = 0
    num_artificial = 0
    num_slack = 0
    c = []
    i = 0
    num_nodes = 0
    while(line.find("end A") == -1):
        rowA = [float(i) for i in line.split(" ")]
        num_nodes = len(rowA)
        for j in range(len(rowA)):
            c.append(rowA[j])
            variables.append("x"+str(i)+str(j))
        i += 1
        line = inp.readline()

    A = []
    b = []

    # sum of xij for ith node = 2, so we take >=2 and <=2
    start = 0
    for i in range(0, num_nodes):
        rowA = [0.0] * len(variables)
        for j in range(start, start+num_nodes):
            rowA[j] = 1.0
        A.append(rowA)
        b.append(2.0)
        num_slack+=1
        for itr in range(0, len(A)):
            A[itr].append(0.0)
        A[-1][-1] = 1.0
        variables.append("S" + str(num_slack))
        initial_basis.append("S" + str(num_slack))
        start += num_nodes

    start = 0
    for i in range(0, num_nodes):
        rowA = [0.0] * len(variables)
        for j in range(start, start+num_nodes):
            rowA[j] = 1.0
        A.append(rowA)
        b.append(2.0)
        num_slack+=1
        num_artificial+=1
        for i in range(0, len(A)):
            A[i].append(0.0)
            A[i].append(0.0)
        # for surplus variable
        A[-1][-1] = 1.0
        A[-1][-2] = -1.0
        variables.append("S" + str(num_slack))
        variables.append("A" + str(num_artificial))
        initial_basis.append("A" + str(num_artificial))
        start += num_nodes
    
    # xij <= 1
    for i in range(0, num_nodes):
        for j in range(0, num_nodes):
            rowA = [0.0] * len(variables)
            rowA[i*num_nodes+j] = 1.0
            b.append(1.0)
            A.append(rowA)
            num_slack+=1
            for itr in range(0, len(A)):
                A[itr].append(0.0)
            A[-1][-1] = 1.0
            variables.append("S" + str(num_slack))
            initial_basis.append("S" + str(num_slack))
    
    # xii = 0, so we take >=0, <=0
    for i in range(0, num_nodes):
        rowA = [0.0] * len(variables)
        rowA[i*num_nodes+i] = 1.0
        b.append(0.0)
        A.append(rowA)
        num_slack+=1
        for itr in range(0, len(A)):
            A[itr].append(0.0)
        A[-1][-1] = 1.0
        variables.append("S" + str(num_slack))
        initial_basis.append("S" + str(num_slack))
    
    for i in range(0, num_nodes):
        rowA = [0.0] * len(variables)
        rowA[i*num_nodes+i] = 1.0
        b.append(0.0)
        A.append(rowA)
        num_slack+=1
        num_artificial+=1
        for i in range(0, len(A)):
            A[i].append(0.0)
            A[i].append(0.0)
        # for surplus variable
        A[-1][-1] = 1.0
        A[-1][-2] = -1.0
        variables.append("S" + str(num_slack))
        variables.append("A" + str(num_artificial))
        initial_basis.append("A" + str(num_artificial))

    # for all subsets, sum of x from S to N\S is >=2

    subsets = powerset(range(0, num_nodes))
    for sub in subsets:
        if len(sub) != 0 and len(sub) != num_nodes:
            not_sub = []
            for val in range(0, num_nodes):
                if val not in sub:
                    not_sub.append(val)
            rowA = [0.0] * len(variables)
            for i in sub:
                for j in not_sub:
                    rowA[i*num_nodes+j] = 1.0
            b.append(2.0)
            A.append(rowA)
            num_slack+=1
            num_artificial+=1
            for i in range(0, len(A)):
                A[i].append(0.0)
                A[i].append(0.0)
            # for surplus variable
            A[-1][-1] = 1.0
            A[-1][-2] = -1.0
            variables.append("S" + str(num_slack))
            variables.append("A" + str(num_artificial))
            initial_basis.append("A" + str(num_artificial))

    num_col_A = len(A[0])
    num_row_A = len(A)

    # putting artificial variables at the end of tableau
    
    k = len(variables) - 1
    for j in range(0, len(A[0])):
        while variables[k].find("A") != -1:
            k-=1
        if variables[j].find("A") != -1 and j < k and k >= 0:
            for i in range(0, len(A)):
                temp = A[i][j]
                A[i][j] = A[i][k]
                A[i][k] = temp 
            temp = variables[j]
            variables[j] = variables[k]
            variables[k] = temp

    # print(variables)
    return A, num_row_A, num_col_A, b, c, num_slack, num_artificial, initial_basis, variables

    
def print_bb_results(message, opt_val, opt_vect, num_x, num_cuts):
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
    ini_A, num_row_A, num_col_A, ini_b, ini_c, ini_num_slack, ini_num_artificial, ini_B, ini_vars = initialize_bb(ip_file)
    # print(A, b, c)
    # print(num_row_A, num_col_A)
    # print(num_slack, num_artificial)
    # print(B, vars)
    flag = 1
    num_splits = 0

    message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix = lp_solve(copy.deepcopy(ini_A), copy.deepcopy(ini_b), copy.deepcopy(ini_c), copy.deepcopy(ini_B), copy.deepcopy(ini_vars), ini_num_artificial, ini_num_slack)
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
                print_bb_results(message, opt_val, opt_val_vector, num_col_A, num_splits)
                sys.stdout.close()
                break
        else:
            sys.stdout = open(op_file, 'w')
            print_bb_results(message, opt_val, opt_val_vector, num_col_A, num_splits)
            sys.stdout.close()
            flag = 0

        # updating A, b, c, B, vars, num_artificial, num_slack 
        num_splits+=1
        
        ini_num_slack+=1
        bs = copy.deepcopy(basic_variables)
        xb = copy.deepcopy(x_B)
        a = copy.deepcopy(A_matrix)

        vars = copy.deepcopy(ini_vars)
        basic_variables = copy.deepcopy(ini_B)
        frac_key_ind = 0
        x_B_left = ini_b
        x_B_right = ini_b

        frac_eq_ind = bs.index(frac_var)

        arti_flag = 0

        # making the split according to the variable value

        for ind in range(0, len(c_j)):
            if vars[ind] == frac_var:
                frac_key_ind = ind
                ini_num_slack += 1
                vars.append("Sb"+str(num_splits))
                basic_variables.append("Sb"+str(num_splits))
                arti_flag = 1
                ini_num_artificial += 1
                vars.append("Ab"+str(num_splits))
                basic_variables.append("Ab"+str(num_splits))
                x_B_left.append(math.floor(xb[frac_eq_ind]))
                x_B_right.append(math.ceil(xb[frac_eq_ind]))
                break

        c_j.append(0.0)

        num_c = 0
        A_matrix = ini_A
        for i in range(0, len(A_matrix)):
            A_matrix[i].append(0.0)
            if arti_flag == 1:
                A_matrix[i].append(0.0)
            num_c = len(A_matrix[i])
        new_constr = [0.0] * num_c

        # ind = 0
        new_constr[frac_key_ind] = 1.0

        new_constr[-1] = 1.0
        if arti_flag == 1:
            new_constr[-2] = -1.0
        A_matrix.append(new_constr)

        # print(num_splits)
        # print("new constr: ", new_constr)
       
        message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix = lp_solve(A_matrix, x_B_left, ini_c, basic_variables, vars, ini_num_artificial, ini_num_slack)

        if message == "Infeasible":
            message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix = lp_solve(A_matrix, x_B_right, ini_c, basic_variables, vars, ini_num_artificial, ini_num_slack)
        # print(message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix)
        # print()



