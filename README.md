# Optimization Methods
Python implementations of Simplex Method, Branch and Bound, Ellipsoid Method and Gomory's Cutting Plane Method.


## Simplex Algorithm
Simplex algorithm is implemented in simplex.py to solve given optimization problem.  Two phase method is used for initialization of simplex algorithm.

<p align="center"><img src="https://latex.codecogs.com/svg.image?\text{min&space;}&space;\mathbf{c}^T\mathbf{x}\\\text{subject&space;to&space;}&space;\textit{A}\mathbf{x}\leq&space;\textbf{b}\\\text{\hspace{5em}}\mathbf{x}\geq&space;0&space;" /> </p>

### Input Format

Each input file for this section has the following structure:

<p align="center"><img src="https://latex.codecogs.com/svg.image?\\\text{start&space;A}&space;\\<&space;\text{row&space;1&space;of&space;matrix}&space;>\\<&space;\text{row&space;2&space;of&space;matrix}&space;>\\\cdots&space;\\<&space;\text{row&space;n&space;of&space;matrix}&space;>\\\text{end&space;A}&space;\\\text{start&space;b}&space;\\<&space;\text{vector&space;b}&space;>\\\text{end&space;b}&space;\\\text{start&space;c}&space;\\&space;<&space;\text{vector&space;c}&space;>\\\text{end&space;c}&space;\\&space;&space;" /> </p>

See [sample_input_simplex.txt](input/simplex/sample_input_simplex.txt) for reference.

### Output Format
- If the optimal solution exists, then the optimal value if given as output along with the vector of optimal values of the variables.
- If there is unbounded solution, outputs ”Unbounded”.
- If infeasible, outputs ”Infeasible”.

See [sample_output_simplex.txt](output/simplex/sample_output_simplex.txt) for reference.

## Branch and Bound

Problem taken: Given a set of villages and distance between every pair of villages, find a tour (a cycle that visits all the nodes) of minimum cost. The cut set formulation of travelling salesman problem is used here. 

Branch and bound algorithm is used to find a tour with minimum cost. To solve LP relaxation problem, the Simplex Algorithm routing developed above is used.

### Input Format

Each input file for this section has the following structure:

<p align="center"><img src="https://latex.codecogs.com/svg.image?\\\text{start&space;A}&space;\\<&space;\text{row&space;1&space;of&space;matrix}&space;>\\<&space;\text{row&space;2&space;of&space;matrix}&space;>\\\cdots&space;\\<&space;\text{row&space;n&space;of&space;matrix}&space;>\\\text{end&space;A}&space;\\" /> </p>

See [sample_input_bb.txt](input/bb/sample_input_bb.txt) for reference.

### Output Format
- If a tour exists: 
  - (a) Outputs the cost of the minimum distance tour.
  - (b) Outputs a Boolean vector of dimension |E|, where i th value denotes whether i th edge is included in the tour or not. 
  - (c) Outputs the number of nodes explored (or output the number of LP relaxations solved).
- If there does not exist any tour, then outputs ”Infeasible Problem”, the number of nodes explored.

See [sample_output_bb.txt](output/bb/sample_output_bb.txt) for reference.

## Cutting Plane Method
Gomory’s cutting plane method is implemented to solve given integer programming problem. To solve LP relaxation problem, the Simplex Algorithm routing developed above is used.

<p align="center"><img src="https://latex.codecogs.com/svg.image?\text{min&space;}&space;\mathbf{c}^T\mathbf{x}\\\text{subject&space;to&space;}&space;\textit{A}\mathbf{x}\leq&space;\textbf{b}\\\text{\hspace{5em}}\mathbf{x}\geq&space;0&space;" /> </p>

### Input format

Each input file for this section has the following structure:

<p align="center"><img src="https://latex.codecogs.com/svg.image?\\\text{start&space;A}&space;\\<&space;\text{row&space;1&space;of&space;matrix}&space;>\\<&space;\text{row&space;2&space;of&space;matrix}&space;>\\\cdots&space;\\<&space;\text{row&space;n&space;of&space;matrix}&space;>\\\text{end&space;A}&space;\\\text{start&space;b}&space;\\<&space;\text{vector&space;b}&space;>\\\text{end&space;b}&space;\\\text{start&space;c}&space;\\&space;<&space;\text{vector&space;c}&space;>\\\text{end&space;c}&space;\\&space;&space;" /> </p>

See [sample_input_cutting_plane.txt](input/cutting_plane/sample_input_cutting_plane.txt) for reference.

### Output format
- If solution exists,
  - (a) Outputs the optimal value (up to 6 decimals). 
  - (b) Outputs an N - dimensional integer vector, denoting optimal solution. 
  - (c) Outputs the number of cutting planes generated (or the number of LP relaxations solved).
- In case of unbounded solution, outputs ”Unbounded” and the number of cutting planes generated (or the number of LP relaxations solved).
- In case of no solution outputs ”Infeasible Solution” and the number of cutting planes generated (or the number of LP relaxations solved).

See [sample_output_cutting_plane.txt](output/cutting_plane/sample_output_cutting_plane.txt) for reference.
