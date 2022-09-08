# Simplex for minimization problem

import numpy as np
import copy
import sys

INPUT_PATH = "./Q1/"
OUTPUT_PATH = "./Q1/"
ip_file = INPUT_PATH +  "input.txt"
op_file = OUTPUT_PATH + "output.txt"

# taking A, b c as input, converting to canonical form in two-phased method
def initialize(ip_file):
    inp = open(ip_file, "r")
    inp.readline()
    line = inp.readline()
    initial_basis = []
    variables = []
    # reading A matrix
    A = []
    num_col_A = -1
    num_row_A = 0
    num_artificial = 0
    num_slack = 0
    while(line.find("end A") == -1):
        rowA = [float(i) for i in line.split(" ")]
        if num_col_A == -1:
            num_col_A = len(rowA)
        num_row_A += 1
        A.append(rowA)
        line = inp.readline()
    # x variables
    for i in range(0, num_col_A):
        variables.append("x"+str(i+1))
    # slack/surplus variables
    for i in range(0, num_row_A):
        variables.append("S"+str(i+1))
        num_slack += 1
    # putting 1.0 value corresponding to slack variable in A
    ind = 0
    for row in A:
        row += [0.0]*num_row_A
        row[num_col_A+ind] = 1.0
        ind += 1
    inp.readline()
    line = inp.readline()
    b = []
    # reading b vector
    while(line.find("end b") == -1):
        ind = 0
        for i in line.split(" "):
            val = float(i)
            b.append(val)
            # if b < 0, we add artificial variables and flip the signs
            if val < 0:
                b[-1] = -b[-1]
                num_artificial += 1
                for row in A:
                    row += [0.0]
                A[ind] = [i * -1.0 for i in A[ind]]
                A[ind][len(A[ind])-1] = 1.0
                initial_basis.append("A"+str(num_artificial))
                variables.append("A"+str(num_artificial))
            else:
                initial_basis.append("S"+str(ind+1))
            ind += 1
        line = inp.readline()
    inp.readline()
    line = inp.readline()
    c = []
    # reading c vector
    while(line.find("end c") == -1):
        for i in line.split(" "):
            c.append(float(i))
        line = inp.readline()
    inp.close()
    return A, num_row_A, num_col_A, b, c, num_slack, num_artificial, initial_basis, variables


# evaluation in each iteration
def simplex_iteration(basic_variables, x_B, c_j, A_matrix, c, vars, itr_num):
    # print("Simplex Iteration Number: ", itr_num)
    # print("A: ", A_matrix)
    # print("B: ", basic_variables)
    # print("x_B: ", x_B)
    # print("c_j: ", c_j)

    c_B = []
    # calculating c_B from c_j
    # print("oho: ", basic_variables, vars)
    for var in basic_variables:
        # print(vars)
        c_B.append(c_j[vars.index(var)])
    # print("c_B: ", c_B)

    # calculating z_j = c_b_j * x_b_j - c_j
    z_j = [0.0] * len(vars)
    for i in range(0, len(z_j)):
        for j in range(0, len(c_B)):
            z_j[i] += (c_B[j] * A_matrix[j][i])
    # print("z_j: ", z_j)

    # calculating z_j-c_j s
    z_j_minus_c_j = [0.0] * len(vars)
    for i in range(0, len(z_j_minus_c_j)):
        z_j_minus_c_j[i] = z_j[i] - c_j[i]
    # print("z_j - c_j: ", z_j_minus_c_j)

    # if all z_j-c_j s <= 0, then we have optimal solution 
    # but if the basic variables have non zero artificial variable, 
    # then the original LP is infeasible
    if all(val <= 0 for val in z_j_minus_c_j):
        message = ""
        inf_flag = 0
        opt_val = 0
        opt_vector = {}
        for i in range(0, len(vars)):
            opt_vector[vars[i]] = 0.0
        for i in range(0, len(basic_variables)):
            if basic_variables[i].find("A") != -1 and x_B[i] != 0:
                inf_flag = 1
        if not inf_flag:
            message = "Optimal"
            for i in range(0, len(basic_variables)):
                opt_vector[basic_variables[i]] = x_B[i]
            for i in range(0, len(c_j)):
                opt_val += (c_j[i] * opt_vector[vars[i]])
        else:
            message = "Infeasible"
        return message, opt_val, opt_vector, basic_variables, x_B, c_j, A_matrix

    entering_variable = ""
    leaving_variable = ""
    
    # taking the positive maxinum in z_j-c_j
    key_col = np.argmax(z_j_minus_c_j)
    entering_variable = vars[key_col]

    # calculating ratios for that column
    ratios = []
    unbounded_flag = 1
    ind = 0
    for row in A_matrix:
        val = 0
        if row[key_col] <= 0:
            val = sys.maxsize
        else:
            val = x_B[ind]/row[key_col]
        ratios.append(val)
        if row[key_col] > 0:
            unbounded_flag = 0
        ind += 1
    
    opt_val = 0
    opt_vector = {}
    message = ""
    if unbounded_flag == 1:
        message = "Unbounded"
        return message, opt_val, opt_vector, basic_variables, x_B, c_j, A_matrix

    # taking minimum ratio
    key_row = np.argmin(ratios)
    # print("ratios: ", ratios)
    leaving_variable = basic_variables[key_row]

    # print("entering variable: ", entering_variable)
    # print("leaving variable: ", leaving_variable)
    
    # replace leaving variable with entering variable
    ind = basic_variables.index(leaving_variable)
    basic_variables = basic_variables[:ind]+[entering_variable]+basic_variables[ind+1:]

    # updating A matrix and x_B
    pivot_value = A_matrix[key_row][key_col]
    num_cols = len(A_matrix[0])
    num_rows = len(A_matrix)

    A_dash = copy.deepcopy(A_matrix)
    x_B_dash = copy.deepcopy(x_B)

    for i in range(0, num_rows):
        for j in range(0, num_cols):
            if i == key_row:
                A_dash[i][j] /= pivot_value
            else:
                A_dash[i][j] -= ((A_matrix[i][key_col] * A_matrix[key_row][j]) / pivot_value)

    for i in range(0, len(x_B)):
        if i == key_row:
            x_B_dash[i] /= pivot_value
        else:
            x_B_dash[i] -= ((x_B[key_row] * A_matrix[i][key_col]) / pivot_value)

    # print("\n")

    # recursive call to next iteration
    return simplex_iteration(basic_variables, x_B_dash, c_j, A_dash, c, vars, itr_num + 1)

    
def print_results(message, opt_val, opt_vect, num_x):
    if message != "Optimal":
        print(message)
        return
    print(round(opt_val, 6))
    x = []
    ind = 1
    for key in opt_vect:
        x.append(opt_vect[key])
        if ind >= num_x:
            break
        ind+=1
    print(*x)


def lp_solve(A_matrix, b, c, B, vars, num_artificial, num_slack):
    opt_val = 0.0
    opt_val_vector = [0.0] * len(vars)
    # initializing basic variables, x_B, c_j
    basic_variables = B
    x_B = b
    c_j = [0.0] * len(vars)
    for i in range(0, len(c_j)):
        if vars[i].find("A") != -1:
            c_j[i] = 1.0

    # Phase 1
    # print("Phase 1 started")
    message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix = simplex_iteration(basic_variables, x_B, c_j, A_matrix, c, vars, itr_num = 1)

    # checking if phase 2 needed 
    if message == "Infeasible" or message == "Unbounded":
        return message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix

    # Phase 2
    # print("Phase 2 started")
    # print(basic_variables)
    # print(x_B)

    # eliminating artificial variables

    # the case where artificial variables are present in basic variables but they are 0
    # if there is any non zero value of non artificial variable in that row, we pivot using that element, else we can safely delete the row

    safely_eliminate = []

    for i in range(0, len(basic_variables)):
        if basic_variables[i].find("A") != -1:
            pivot_flag = 0
            for j in range(0, len(A_matrix[i])):
                if A_matrix[i][j] != 0 and vars[j].find("A") == -1:
                    leaving_variable = basic_variables[i]
                    entering_variable = vars[j]
                    pivot_flag = 1
                    key_row = i
                    key_col = j
                    pivot_value = A_matrix[key_row][key_col]
                    num_cols = len(A_matrix[0])
                    num_rows = len(A_matrix)

                    A_dash = copy.deepcopy(A_matrix)
                    x_B_dash = copy.deepcopy(x_B)

                    for i in range(0, num_rows):
                        for j in range(0, num_cols):
                            if i == key_row:
                                A_dash[i][j] /= pivot_value
                            else:
                                A_dash[i][j] -= ((A_matrix[i][key_col] * A_matrix[key_row][j]) / pivot_value)

                    for i in range(0, len(x_B)):
                        if i == key_row:
                            x_B_dash[i] /= pivot_value
                        else:
                            x_B_dash[i] -= ((x_B[key_row] * A_matrix[i][key_col]) / pivot_value)
                    A_matrix = A_dash
                    x_B = x_B_dash
                    ind = basic_variables.index(leaving_variable)
                    basic_variables = basic_variables[:ind]+[entering_variable]+basic_variables[ind+1:]
                    break
            if pivot_flag == 0:
                safely_eliminate.append(i)

    for index in sorted(safely_eliminate, reverse=True):
        del A_matrix[index]
        del basic_variables[index]

    for i in range(0, len(A_matrix)):
        j = 0
        while(j < num_artificial):
            A_matrix[i].pop()
            j += 1
    j = 0
    while(j < num_artificial):
        vars.pop()
        j += 1
    
    c_j = [0.0] * len(vars)
    for i in range(0, len(c)):
        c_j[i] = c[i]

    # print("oho1: ", basic_variables, vars)
    message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix = simplex_iteration(basic_variables, x_B, c_j, A_matrix, c, vars, itr_num = 1)
    return message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix

if __name__ == "__main__":
    if len(sys.argv)>1:
        ip_file = INPUT_PATH + sys.argv[1]
        op_file = OUTPUT_PATH + sys.argv[1]
    A, num_row_A, num_col_A, b, c, num_slack, num_artificial, B, vars = initialize(ip_file)
    # print(A, b, c)
    # print(num_row_A, num_col_A)
    # print(num_slack, num_artificial)
    # print(B, vars)
    message, opt_val, opt_val_vector, basic_variables, x_B, c_j, A_matrix = lp_solve(A, b, c, B, vars, num_artificial, num_slack)
    sys.stdout = open(op_file, 'w')
    print_results(message, opt_val, opt_val_vector, num_col_A)
    sys.stdout.close()
