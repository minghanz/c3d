import time
import torch

class TimingSingerLayer:
    def __init__(self, layer):
        self.cur_name = None
        self.cur_time = None
        self.layer = layer

    def log(self, name, cur_time=None):
        last_name = self.cur_name
        last_time = self.cur_time
        self.cur_name = name
        self.cur_time = time.time() if cur_time is None else cur_time
        if last_time is not None:
            print(' '*int(self.layer)*2 + '{}: {} -> {}: {:.5}'.format(self.layer, last_name, self.cur_name, self.cur_time - last_time))
        else: 
            print('Initialized:', self.cur_name)

class Timing:
    def __init__(self):
        self.timelist = []
        self.timetemp = {}

    def log(self, name, layer, cuda_sync=False):
        if cuda_sync:
            torch.cuda.synchronize()

        init_stage = len(self.timelist) == 0

        while layer >= len(self.timelist):
            self.timelist.append( TimingSingerLayer(len(self.timelist)) )
            if len(self.timelist) == 1:
                self.timelist[len(self.timelist)-1].log(name, cuda_sync)
            else:
                self.timelist[len(self.timelist)-1].log(self.timelist[len(self.timelist)-2].cur_name, self.timelist[len(self.timelist)-2].cur_time)

        if not init_stage:
            cur_time = time.time()
            for cl in range(len(self.timelist)-1, layer-1, -1):
                self.timelist[cl].log(name, cur_time)

    def log_temp(self, name, cuda_sync=False):
        if cuda_sync:
            torch.cuda.synchronize()

        self.timetemp[name] = time.time()
    def log_temp_end(self, name, cuda_sync=False):
        assert name in self.timetemp
        if cuda_sync:
            torch.cuda.synchronize()

        print('{}: {:.5f}'.format(name, time.time() - self.timetemp[name]))
        del self.timetemp[name]

    def log_cuda_sync(self):
        self.log_temp('sync')
        torch.cuda.synchronize()
        self.log_temp_end('sync')

    # def log_init(self, name):
    #     self.start_name = name
    #     self.start_time = time.time()
    #     self.cur_name = self.start_name
    #     self.cur_time = self.start_time
    #     print('Initialized:', self.cur_name)

    # def log_end(self, name):
    #     self.last_name = self.cur_name
    #     self.last_time = self.cur_time
    #     self.cur_name = name
    #     self.cur_time = time.time()
    #     self.end_name = self.cur_name
    #     self.end_time = self.cur_time
    #     if self.last_time is not None:
    #         print('{} to {}:'.format(self.last_name, self.cur_name), self.cur_time - self.last_time)
    #         print('start to end:', self.end_time - self.start_time)