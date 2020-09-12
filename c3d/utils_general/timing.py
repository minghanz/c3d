import time
import torch
import logging

class TimingSingerLayer:
    def __init__(self, layer, logging_mode=False):
        self.cur_name = None
        self.cur_time = None
        self.layer = layer
        self.logging_mode = logging_mode

    def log(self, name, cur_time=None, silent=False):
        last_name = self.cur_name
        last_time = self.cur_time
        self.cur_name = name
        self.cur_time = time.time() if cur_time is None else cur_time
        if not silent:
            if last_time is not None:
                info = ' '*int(self.layer)*2 + '{}: {} -> {}: {:.5}'.format(self.layer, last_name, self.cur_name, self.cur_time - last_time)
            else: 
                info = 'Initialized: '+ self.cur_name

            if self.logging_mode:
                logging.info(info)
            else:
                print(info)

class Timing:
    def __init__(self, logging_mode=False):
        self.timelist = []
        self.timetemp = {}
        self.logging_mode = logging_mode

    def log(self, name, layer, cuda_sync=False):
        if cuda_sync:
            torch.cuda.synchronize()

        init_stage = len(self.timelist) == 0

        ### Initialize new layer(s). For higher layers, set the initial log using the lower layer's cur_time. 
        ### If originally no time layers (init_stage==True), set the first layer with the current time. It corresponds to the very first log, and only happens once.
        while layer >= len(self.timelist):
            self.timelist.append( TimingSingerLayer(len(self.timelist), self.logging_mode) )
            if len(self.timelist) == 1:
                self.timelist[len(self.timelist)-1].log(name)
            else:
                self.timelist[len(self.timelist)-1].log(self.timelist[len(self.timelist)-2].cur_name, self.timelist[len(self.timelist)-2].cur_time)

        ### If originally there are some time layers (init_stage==False), log the current time to the specified layer and all higher layers. 
        if not init_stage:
            cur_time = time.time()
            for cl in range(len(self.timelist)-1, layer-1, -1):
                self.timelist[cl].log(name, cur_time, silent=cl>layer)

    def log_temp(self, name, cuda_sync=False):
        if cuda_sync:
            torch.cuda.synchronize()

        self.timetemp[name] = time.time()
    def log_temp_end(self, name, cuda_sync=False):
        assert name in self.timetemp
        if cuda_sync:
            torch.cuda.synchronize()

        info = '{}: {:.5f}'.format(name, time.time() - self.timetemp[name])
        if self.logging_mode:
            logging.info(info)
        else:
            print(info)
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