import numpy as np
import sys

grid_size = 0
gamma = 0
noise = []
board = []
newboard = []
used_indices = []
original_indices = []
living_reward = 0

def getInput():
    global grid_size, gamma, noise, board, newboard
    with open("C:/Users/super/All CS Work/ML Class Work/9 Markov Decision Process/final.txt", "r") as my_file:
        count = 0
        for line in my_file:
            if(count == 0):
                grid_size = int(line.strip())
            if(count == 1):
                gamma = float(line.strip())
            if(count == 2):
                noise = line.strip().split(", ")
                noise = [float(i) for i in noise]
                if(len(noise) < 4):
                    noise.append(0)
            if(count > 3):
                tempArray = np.array(line.strip().split(","))
                if(count == 4):
                    board = tempArray
                    newboard = tempArray
                else:
                    board = np.concatenate((board, tempArray), axis = 0)
                    newboard = np.concatenate((newboard, tempArray), axis = 0)
            count += 1
        newboard = newboard.tolist()

    print("grid size", grid_size)
    print("gamma", gamma)
    print("noise", noise)
    print("board: ", "\n", board)

def find_mdp_start():
    global board, gamma, noise, grid_size, used_indices, original_indices
    max = float('-inf')
    max_index = 0
    for idx, x in np.ndenumerate(board):
        # print(idx, x)
        if(x != 'X'):
            temp = float(x)
            original_indices.append(idx[0])
            if(temp > max):
                max = temp
                max_index = idx
    return max_index

def print_board(b):
    global grid_size, used_indices
    count = 1
    for i in range(len(b)):
        if(count % grid_size == 0):
            print(str(b[i]) + "\n", end = "")
        else:
            spaces = 7 - len(str(b[i]))
            print(str(b[i]), end = " " * spaces)
        count+=1

def run_mdp_scratch(index):
    global grid_size, gamma, noise, board, living_reward, newboard
    used_indices.append(index)
    # print("index: ", index)
    # print_board(newboard)
    if(newboard[index] == 'X' and index not in original_indices):
        up_index = index - grid_size
        down_index = index + grid_size
        left_index = index - 1
        right_index = index + 1

        if(up_index >= 0):
            up_val = newboard[up_index]
        else:
            up_val = "X"
        if(down_index < len(board)):
            down_val = newboard[down_index]
        else:
            down_val = "X"
        if(int(left_index/grid_size) == int(index/grid_size)):
            left_val = newboard[left_index]
        else:
            left_val = "X"
        if(int(right_index/grid_size) == int(index/grid_size)):
            right_val = newboard[right_index]
        else:
            right_val = "X"

        #north
        n_val = 0
        if(up_val != "X"):
            n_val += noise[0] * (living_reward + (gamma * float(up_val)))
        if(right_val != "X"):
            n_val += noise[1] * (living_reward + (gamma * float(right_val)))
        if(left_val != "X"):
            n_val += noise[2] * (living_reward + (gamma * float(left_val)))
        if(down_val != "X"):
            n_val += noise[3] * (living_reward + (gamma * float(down_val)))

        #east
        e_val = 0
        if(right_val != "X"):
            e_val += noise[0] * (living_reward + (gamma * float(right_val)))
        if(down_val != "X"):
            e_val += noise[1] * (living_reward + (gamma * float(down_val)))
        if(up_val != "X"): 
            e_val += noise[2] * (living_reward + (gamma * float(up_val)))
        if(left_val != "X"):
            e_val += noise[3] * (living_reward + (gamma * float(left_val)))

        w_val = 0
        if(left_val != "X"):
            w_val += noise[0] * (living_reward + (gamma * float(left_val)))
        if(up_val != "X"):
            w_val += noise[1] * (living_reward + (gamma * float(up_val)))
        if(down_val != "X"): 
            w_val += noise[2] * (living_reward + (gamma * float(down_val)))
        if(right_val != "X"):
            w_val += noise[3] * (living_reward + (gamma * float(right_val)))

        s_val = 0
        if(down_val != "X"):
            s_val += noise[0] * (living_reward + (gamma * float(down_val)))
        if(left_val != "X"):
            s_val += noise[1] * (living_reward + (gamma * float(left_val)))
        if(right_val != "X"): 
            s_val += noise[2] * (living_reward + (gamma * float(right_val)))
        if(up_val != "X"):
            s_val += noise[3] * (living_reward + (gamma * float(up_val)))


        # print("Index", index, "vals N-E-W-S:", n_val, e_val, w_val, s_val)
        maximum = max((n_val, e_val, w_val, s_val))
        newboard[index] = round(maximum, 2)
        # sys.exit(0)
    #left:
    if(index-1 >= 0 and int((index - 1)/grid_size) == int(index/grid_size) and index-1 not in used_indices):
        run_mdp_scratch(index-1)

    #right:
    if(index + 1 < len(board) and int((index + 1)/grid_size) == int(index/grid_size) and index+1 not in used_indices):
        run_mdp_scratch(index+1)

    #up:
    if(index-grid_size >= 0 and int(index - grid_size) > 0 and index-grid_size not in used_indices):
        run_mdp_scratch(index-grid_size)

    #down:
    if(index+grid_size < len(board) and int(index + grid_size) < len(board) and index+grid_size not in used_indices):
        run_mdp_scratch(index+grid_size)

def run_mdp_iter(index):
    global grid_size, gamma, noise, board, living_reward, newboard, original_indices
    # print("index: ", index)
    # print_board(newboard)
    if(index not in used_indices and index not in original_indices):
        up_index = index - grid_size
        down_index = index + grid_size
        left_index = index - 1
        right_index = index + 1

        if(up_index >= 0):
            up_val = newboard[up_index]
        else:
            up_val = "X"
        if(down_index < len(board)):
            down_val = newboard[down_index]
        else:
            down_val = "X"
        if(int(left_index/grid_size) == int(index/grid_size)):
            left_val = newboard[left_index]
        else:
            left_val = "X"
        if(int(right_index/grid_size) == int(index/grid_size)):
            right_val = newboard[right_index]
        else:
            right_val = "X"

        #north
        n_val = 0
        if(up_val != "X"):
            n_val += noise[0] * (living_reward + (gamma * float(up_val)))
        if(right_val != "X"):
            n_val += noise[1] * (living_reward + (gamma * float(right_val)))
        if(left_val != "X"):
            n_val += noise[2] * (living_reward + (gamma * float(left_val)))
        if(down_val != "X"):
            n_val += noise[3] * (living_reward + (gamma * float(down_val)))

        #east
        e_val = 0
        if(right_val != "X"):
            e_val += noise[0] * (living_reward + (gamma * float(right_val)))
        if(down_val != "X"):
            e_val += noise[1] * (living_reward + (gamma * float(down_val)))
        if(up_val != "X"): 
            e_val += noise[2] * (living_reward + (gamma * float(up_val)))
        if(left_val != "X"):
            e_val += noise[3] * (living_reward + (gamma * float(left_val)))

        w_val = 0
        if(left_val != "X"):
            w_val += noise[0] * (living_reward + (gamma * float(left_val)))
        if(up_val != "X"):
            w_val += noise[1] * (living_reward + (gamma * float(up_val)))
        if(down_val != "X"): 
            w_val += noise[2] * (living_reward + (gamma * float(down_val)))
        if(right_val != "X"):
            w_val += noise[3] * (living_reward + (gamma * float(right_val)))

        s_val = 0
        if(down_val != "X"):
            s_val += noise[0] * (living_reward + (gamma * float(down_val)))
        if(left_val != "X"):
            s_val += noise[1] * (living_reward + (gamma * float(left_val)))
        if(right_val != "X"): 
            s_val += noise[2] * (living_reward + (gamma * float(right_val)))
        if(up_val != "X"):
            s_val += noise[3] * (living_reward + (gamma * float(up_val)))


        # print("Index", index, "vals N-E-W-S:", n_val, e_val, w_val, s_val)
        maximum = max((n_val, e_val, w_val, s_val))
        newboard[index] = round(maximum, 2)
        # sys.exit(0)
    used_indices.append(index)
    #left:
    if(index-1 >= 0 and int((index - 1)/grid_size) == int(index/grid_size) and index-1 not in used_indices):
        run_mdp_iter(index-1)

    #right:
    if(index + 1 < len(board) and int((index + 1)/grid_size) == int(index/grid_size) and index+1 not in used_indices):
        run_mdp_iter(index+1)

    #up:
    if(index-grid_size >= 0 and int(index - grid_size) > 0 and index-grid_size not in used_indices):
        run_mdp_iter(index-grid_size)

    #down:
    if(index+grid_size < len(board) and int(index + grid_size) < len(board) and index+grid_size not in used_indices):
        run_mdp_iter(index+grid_size)
    
getInput()
mdp_start = find_mdp_start()
print("Original Indices: ", original_indices)
print(original_indices[0])
print_board(board)
run_mdp_scratch(mdp_start[0])
used_indices = []
print("Final Board from 1st Run: ")
print_board(newboard)
runs = int(input("How Many Iterations?"))
for i in range(runs):
    run_mdp_iter(mdp_start[0])
    print("Final Board from " + str(i) + " Iteration: ")
    print_board(newboard)