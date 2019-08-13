import numpy as np


_dir = 'E:\challenge\SNER\solutions\\'


def read_solution_file(_file):
    return np.loadtxt(_dir + _file)


def write_solution(_solution_path, data):
    with open(_solution_path, 'w') as file:
        for line in data:
            file.write("{}\n".format(line))


f1 = read_solution_file('solutions_s1.txt')
f2 = read_solution_file('solutions_s2.txt')
f3 = read_solution_file('solutions_s3.txt')
f4 = read_solution_file('solutions_s4.txt')
f5 = read_solution_file('solutions_s5.txt')
f6 = read_solution_file('solutions_s6.txt')


m = (f1+f2+f3+f4+f5+f6)/6
# m = (f7+f8)/2
solution_path = _dir+'ensemble_s1.txt'

write_solution(solution_path, m)

i = 1
