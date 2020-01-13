import numpy as np
import math
import tensorflow as tf

class PostDTW:
    def __init__(self):
        self.cost = {}
        self.trace = {}
        self.step = np.array([[-1,0], [-1,-1], [0,-1]])
        self.maxDelay = 5#int(2/0.032)
        self.refN = 0
        self.playN = 0
        self.refFrames = []
        self.playFrames = []
        self.tunedFrames = []
        self.vadlidF = 0
        self.PUNISH = [1.5, 0, 1.5]

    def note_cost(self, key1, key2):
        cost = 0
        
        for i in range(key1.shape[0]):
            cost += abs(key1[i] - key2[i])
            '''
        k1 =  tf.convert_to_tensor(key1, dtype=tf.float32)
        k2 =  tf.convert_to_tensor(key2, dtype=tf.float32)
        c = tf.reduce_sum(tf.abs(tf.subtract(k1, k2)))
        cost = c.eval()
        '''
        return cost

    def finetune_prob(self, played, refer):
        scale = np.ones(88, dtype=np.float)
        for key in range(refer.shape[0]):
            if refer[key] <= 0:
                continue
            # increase the probability of keys of refer score
            scale[key] = 5
            # set probability to 0 for harmonics of refer notes
            for i in range(10):
                for j in range(-1, 3, 2):
                    k = key + 12 * i * j
                    if k < 0 or k >= 88:
                        continue
                    if refer[k] > 0:
                        continue
                    scale[k] = 0
        prob = np.zeros(88, dtype=np.float)
        # re-scale the probability according to score
        for key in range(played.shape[0]):
            prob[key] = played[key] * scale[key]
        return prob

    def align_play_on_score_offline(self, played, ref):
        [playNum, cols] = played.shape
        for i in range(playNum):
            played[i] = self.finetune_prob(played[i], ref[i])
        return played

    def align_play_on_score(self, played, ref):
        [playNum, cols] = played.shape
        [refNum, cols] = ref.shape

        if self.playN == 0:
            self.refFrames = ref
            self.playFrames= played
        else:
            self.refFrames = np.vstack((self.refFrames, ref))
            self.playFrames= np.vstack((self.playFrames, played))

        if self.playN == 0:
            self.cost[(0,0)] = self.note_cost(played[0], ref[0])

        startF = endF = 0
        x = y = 0
        for i in range(playNum):
            if self.playN == 0 and i==0:
                continue
            x = self.playN + i
            startF = x - self.maxDelay
            startF = max(startF, 0)
            endF = x + self.maxDelay
            endF = min(endF, self.refFrames.shape[0])
            tryRange = list(range(self.vadlidF, startF-1, -1)) + list(range(self.vadlidF+1, endF))
            print('[', i, ']:', startF, ' - ', endF)

            for y in tryRange:
                cost = self.note_cost(played[i], self.refFrames[y])
                mincost = 1E38
                idx = -1
                for k in range(self.step.shape[0]):
                    prex = x + self.step[k][0]
                    prey = y + self.step[k][1]
                    if (prex, prey) not in self.cost:
                        continue
                    cost_sum = self.cost[prex,prey] + self.PUNISH[k]
                    if cost_sum < mincost:
                        mincost = cost_sum
                        idx = k
                if idx == -1:
                    continue

                self.cost[(x,y)] = mincost + cost
                self.trace[(x,y)] = idx

        mincost = 1E38
        idx = -1
        for y in range(startF, endF):
            if (x,y) not in self.cost:
                continue
            if self.cost[(x, y)] < mincost:
                mincost = self.cost[(x,y)]
                idx = y

        if idx == -1:
            print("something is wrong!")
            exit(-1)

        y = idx
        tuned = self.finetune_prob(self.playFrames[x], self.refFrames[y])
        self.tunedFrames = tuned
        print("trace back path")
        while x > self.playN:
            direct = self.trace[x, y]
            x = x + self.step[direct][0]
            y = y + self.step[direct][1]
            tuned = self.finetune_prob(self.playFrames[x], self.refFrames[y])
            if self.step[direct][0] < 0:
                self.tunedFrames = np.vstack((tuned, self.tunedFrames))
            #print(x, y)

        self.playN = self.playFrames.shape[0]
        self.refN  = self.refFrames.shape[0]
        self.vadlidF = idx
        return  self.tunedFrames

