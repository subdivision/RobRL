import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from collections import defaultdict
from PIL import Image

NUM_OF_ITERS = 100
ROWS = 400
COLS = 400
VISITED = 10
OBSTACLE = 0

#-----------------------------------------------------------------------------
class OneLayerGraph:

    #-----------------------------------------------------------------------------
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.deltas = [ (1, 0), (0, 1), (-1, 0), (0, -1) ]
        self.init_maze()

    #-----------------------------------------------------------------------------
    def init_maze(self):
        self.matrix = np.ones( (self.rows, self.cols), dtype = np.int8)
        # "draw" some obstacles
        #self.matrix[self.rows//2, self.cols//4:self.cols*3//4] = OBSTACLE
        self.matrix[self.rows//2, 0:self.cols*3//4] = OBSTACLE
        self.matrix[self.rows//4:self.rows//2+1, self.cols*3//4] = OBSTACLE

    #-----------------------------------------------------------------------------
    def init_reward_qvals(self, target_idx = (99,99)):
        self.reward = np.zeros( (self.rows*2+1, self.cols*2+1), dtype = np.float)
        self.qvals  = np.zeros( (self.rows*2+1, self.cols*2+1), dtype = np.float)

        for neig_idx in self.get_valid_edge_deltas(target_idx, without_obstacles=True):
            self.reward[ target_idx[0] * 2 + 1 + neig_idx[0], target_idx[1] * 2 + 1 + neig_idx[1] ] = 100

        self.qvals[0,:] = -100
        self.qvals[:,0] = -100
        self.qvals[self.rows*2, :] = -100
        self.qvals[:, self.cols*2] = -100
        for i in range(self.rows):
            for j in range(self.cols):
                curr_to_neigh_qval = -100 if self.matrix[i,j] <= OBSTACLE else 0
                for d in self.get_valid_edge_deltas((i,j), without_obstacles=False):
                    neigh_to_curr_qval = -100 if self.matrix[i+d[0], j+d[1]] <= OBSTACLE else 0
                    qval = min( curr_to_neigh_qval, neigh_to_curr_qval )
                    self.qvals[ i*2+1+d[0], j*2+1+d[1] ] = qval

    #-----------------------------------------------------------------------------
    def get_valid_edge_deltas(self, curr_idx, without_obstacles = True):
        result = []
        i = curr_idx[0]*2+1
        j = curr_idx[1]*2+1
        for d in self.deltas:
            vrtx_i = curr_idx[0] + d[0]
            if 0 > vrtx_i or vrtx_i >= self.rows:
                continue
            vrtx_j = curr_idx[1] + d[1]
            if 0 > vrtx_j or vrtx_j >= self.cols:
                continue
            if self.matrix[vrtx_i, vrtx_j] <= OBSTACLE and without_obstacles:
                continue
            edge_i = i+d[0]
            edge_j = j+d[1]
            if 0 == edge_i or self.rows*2 == edge_i or 0 == edge_j or self.cols*2+1 == edge_j:
                #fictional edges of the boundary
                continue
            result.append( d )
        return result

    #-----------------------------------------------------------------------------
    def get_all_deltas_with_max_qval(self, curr_idx):
        candidates = []
        max_reward = -100
        for d in self.get_valid_edge_deltas(curr_idx):
            i = curr_idx[0] * 2 + 1 + d[0]
            j = curr_idx[1] * 2 + 1 + d[1]
            if self.reward[ i, j ] == max_reward:
                candidates.append( d )
            elif self.reward[ i, j ] > max_reward:
                max_reward = self.reward[ i, j ]
                candidates = [ d ]
        return candidates

    #-----------------------------------------------------------------------------
    def next_stop(self, curr_idx, err):
        random_val = random.uniform(0,1)
        candidates = []
        if random_val < err:
            candidates = self.get_valid_edge_deltas(curr_idx)
        else:
            candidates = self.get_all_deltas_with_max_qval(curr_idx)
        next_delta = candidates[ np.random.randint( len(candidates) ) ]
        next_node = ( curr_idx[0] + next_delta[0], curr_idx[1] + next_delta[1] )
        return next_node

    #-----------------------------------------------------------------------------
    def update_qvals(self, curr_idx, next_idx, learning_rate, discount):
        candidates = self.get_all_deltas_with_max_qval(next_idx)
        max_delta = candidates[ np.random.randint( len(candidates) ) ]
        max_value = self.qvals[ next_idx[0]*2+1+max_delta[0], next_idx[1]*2+1+max_delta[1] ]
        delta_r = next_idx[0] - curr_idx[0]
        delta_c = next_idx[1] - curr_idx[1]
        i = next_idx[0] * 2 + 1 + delta_r
        j = next_idx[1] * 2 + 1 + delta_c
        self.qvals[i,j] = int(   (1.-learning_rate) * self.qvals[i, j] \
                               +     learning_rate  * (   self.reward[i, j] \
                                                        + discount * max_value ) )

    #-----------------------------------------------------------------------------
    def get_random_idx(self):
        while True:
            curr_idx = ( np.random.randint( self.rows ), np.random.randint( self.cols ) )
            if self.matrix[curr_idx[0], curr_idx[1]] != OBSTACLE:
                return curr_idx

    #-----------------------------------------------------------------------------
    def learn(self, err, target_idx, learning_rate, discount):
        self.init_reward_qvals(target_idx)
        for i in range(NUM_OF_ITERS):
            curr_idx = self.get_random_idx()
            next_idx = self.next_stop(curr_idx, err )
            self.update_qvals( curr_idx, next_idx, learning_rate, discount )

    #-----------------------------------------------------------------------------
    def get_not_visited_neigh(self, curr_idx, end, deltas):
        dx_min = self.rows
        dy_min = self.cols
        closest_neigh = None
        for d in deltas:
            i = curr_idx[0] + d[0]
            j = curr_idx[1] + d[1]
            if self.matrix[i, j] != VISITED:
                dx_curr = np.abs( end[0] - i )
                dy_curr = np.abs( end[1] - j )
                if dx_curr < dx_min and dy_curr < dy_min:
                    dx_min = dx_curr
                    dy_min = dy_curr
                    closest_neigh = (i, j)
        return closest_neigh

    #-----------------------------------------------------------------------------
    def compute_path(self, begin, end):
        self.matrix[begin[0], begin[1]] = VISITED
        next_idx = begin
        while next_idx != end:
            next_idx = self.get_not_visited_neigh( next_idx, end, self.get_all_deltas_with_max_qval( next_idx ) )
            try:
                self.matrix[next_idx[0],next_idx[1]] = VISITED
            except:
                print (next_idx)

    #-----------------------------------------------------------------------------
    def get_image(self):
        data = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        white = [255, 255, 255]
        black = [  0,   0,   0]
        red   = [255,   0,   0]
        data[self.matrix == OBSTACLE] = black
        data[self.matrix == 1] = white
        data[self.matrix == VISITED] = red
        img = Image.fromarray(data, 'RGB')
        return img

#-----------------------------------------------------------------------------
def main():
    strt = datetime.now()
    print ('Starting at {}'.format( strt.strftime( '%Y-%m-%d %H:%M:%S')))
    begin = (0,0)
    finish = (ROWS-1, COLS-1)
    gr = OneLayerGraph(ROWS, COLS)
    gr.learn(0.8, finish, 0.8, 0.8)
    gr.compute_path(begin, finish)
    fnsh = datetime.now()
    print(str( fnsh-strt ))
    img = gr.get_image()
    img.show()

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()

#================================= END OF FILE ===============================