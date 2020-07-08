import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from collections import defaultdict
from PIL import Image

NUM_OF_ITERS = 4000000
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
        self.deltas = [(1, 0), (1, 1), (0, 1),
                       (-1, 1), (1, -1), (-1, 0),
                       (0, -1), (-1, -1)]
        self.init_maze()

    #-----------------------------------------------------------------------------
    def init_maze(self):
        self.matrix = np.ones( (self.rows, self.cols), dtype = np.int8)
        # "draw" some obstacles
        self.matrix[self.rows//2, self.cols//4:self.cols*3//4] = OBSTACLE
        self.matrix[self.rows//4:self.rows//2+1, self.cols*3//4] = OBSTACLE

    #-----------------------------------------------------------------------------
    def init_reward_qvals(self, target_idx = (99,99)):
        self.reward = defaultdict(float)
        self.qvals = defaultdict(float)

        for src_id in self.get_valid_neighb_indeces(target_idx, without_obstacles=True):
            self.reward[ (src_id, target_idx) ] = 100


        for i in range(self.rows):
            for j in range(self.cols):
                curr_idx = (i,j)
                curr_to_neigh_qval = -100 if self.matrix[i,j] <= 0 else 0
                for neigh_idx in self.get_valid_neighb_indeces(curr_idx, without_obstacles=False):
                    neigh_to_curr_qval = -100 if self.matrix[neigh_idx[0], neigh_idx[1]] <= 0 else 0
                    qval = min( curr_to_neigh_qval, neigh_to_curr_qval )
                    self.qvals[ (curr_idx, neigh_idx) ] = qval
                    self.qvals[ (neigh_idx, curr_idx) ] = qval

    #-----------------------------------------------------------------------------
    def get_valid_neighb_indeces(self, curr_idx, without_obstacles = True):
        result = []
        for d in self.deltas:
            candidate = ( curr_idx[0] + d[0], curr_idx[1] + d[1] )
            if 0 <= candidate[0] < self.rows \
               and 0 <= candidate[1] < self.cols:
                if self.matrix[candidate[0], candidate[1]] <= OBSTACLE and without_obstacles:
                    continue
                result.append( candidate )
        return result

    #-----------------------------------------------------------------------------
    def get_all_neighb_with_max_qval(self, curr_idx):
        candidates = []
        max_reward = -100
        for next_idx in self.get_valid_neighb_indeces(curr_idx):
            if self.reward[(curr_idx, next_idx)] == max_reward:
                candidates.append(next_idx)
            elif self.reward[(curr_idx, next_idx)] > max_reward:
                max_reward = self.reward[(curr_idx, next_idx)]
                candidates = [next_idx]
        return candidates

    #-----------------------------------------------------------------------------
    def next_stop(self, curr_idx, err):
        random_val = random.uniform(0,1)
        candidates = []
        if random_val < err:
            candidates = self.get_valid_neighb_indeces(curr_idx)
        else:
            candidates = self.get_all_neighb_with_max_qval(curr_idx)
        next_node = candidates[ np.random.randint( len(candidates) ) ]
        return next_node

    #-----------------------------------------------------------------------------
    def update_qvals(self, curr_idx, next_idx, learning_rate, discount):
        candidates = self.get_all_neighb_with_max_qval(next_idx)
        max_index = candidates[ np.random.randint( len(candidates) ) ]
        max_value = self.qvals[ (next_idx, max_index) ]
        self.qvals[(curr_idx, next_idx)] = int(   (1.-learning_rate) * self.qvals[(curr_idx, next_idx)] \
                                                +     learning_rate  * ( self.reward[(curr_idx, next_idx)] \
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
    def get_not_visited_neigh(self, end, neighs):
        dx_min = self.rows
        dy_min = self.cols
        closest_neigh = None
        for n in neighs:
            if self.matrix[n[0], n[1]] != VISITED:
                dx_curr = np.abs( end[0] - n[0] )
                dy_curr = np.abs( end[1] - n[1] )
                if dx_curr < dx_min and dy_curr < dy_min:
                    dx_min = dx_curr
                    dy_min = dy_curr
                    closest_neigh = n
                return closest_neigh

    #-----------------------------------------------------------------------------
    def compute_path(self, begin, end):
        self.matrix[begin[0], begin[1]] = VISITED
        next_idx = begin
        while next_idx != end:
            next_idx = self.get_not_visited_neigh( end, self.get_all_neighb_with_max_qval( next_idx ) )
            self.matrix[next_idx[0],next_idx[1]] = VISITED

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
    gr.learn(0.2, finish, 0.8, 0.8)
    gr.compute_path(begin, finish)
    fnsh = datetime.now()
    print(str( fnsh-strt ))
    img = gr.get_image()
    img.save('roboth_path.png')
    img.show()

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()

#================================= END OF FILE ===============================