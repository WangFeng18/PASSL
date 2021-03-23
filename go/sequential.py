import os
import re
import yaml
import time
import hashlib
import argparse

def md5(file_name):
    with open(file_name, 'rb') as fp:
        data = fp.read()
    file_md5= hashlib.md5(data).hexdigest()
    return file_md5

class GroupBunch(object):
    def __init__(self, cfg_file, test=False):
        with open(cfg_file, 'r') as fopen:
            self.cfg = yaml.load(fopen, Loader=yaml.SafeLoader)
        self.exp_name = self.cfg['name']
        self.group_dir = self.cfg['group_dir']
        self.test = test
    
    def forward(self):
        for bunch in self.cfg['group']:
            bunchRunner = BunchTask(bunch, self.group_dir, self.test)
            bunchRunner.forward()

class BunchTask(object):
    def __init__(self, cfg, group_dir, test):
        self.cfg = cfg
        self.task = self.cfg['task']
        self.comments = self.cfg['comments']
        self.pretrain_config = self.cfg['pretrain_config']
        self.linear_config = self.cfg['linear_config']
        self.pretrain_config_base = os.path.basename(self.pretrain_config).split('.')[0]
        self.linear_config_base = os.path.basename(self.linear_config).split('.')[0]
        self.pretrain_md5 = md5(self.pretrain_config)[:8]
        self.linear_md5 = md5(self.linear_config)[:8]
        self.group_dir = group_dir
        self.test = test

    def forward(self):
        for task in self.cfg['bunch_tasks']:
            getattr(self, 'run_{}'.format(task['name']))(task)

    def run_pretrain(self, cfg):
        cudaenv = cfg['CUDA_VISIBLE_DEVICES']
        self.pretrain_timestamp = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        self.pretrain_output_dir = self.parse(cfg['output_dir'])
        cmd = 'CUDA_VISIBLE_DEVICES={} python3 -c {} --num-gpus {} --output_dir {}' \
              .format(cudaenv, self.pretrain_config, cfg['num-gpus'], self.pretrain_output_dir)
        print(cmd)
        if not self.test:
            os.system(cmd)
    
    def run_linear(self, cfg):
        cudaenv = cfg['CUDA_VISIBLE_DEVICES']
        self.linear_timestamp = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        self.linear_pretrain = self.parse(cfg['load'])
        self.linear_output_dir = self.parse(cfg['output_dir'])
        cmd = 'CUDA_VISIBLE_DEVICES={} python3 -c {} --num-gpus {} --output_dir {} --load {}' \
              .format(cudaenv, self.linear_config, cfg['num-gpus'], self.linear_output_dir, self.linear_pretrain)
        print(cmd)
        if not self.test:
            os.system(cmd)

    def parse(self, str):
        res = re.sub(r'{(.*?)}', lambda x: getattr(self, x.group(1)), str)
        print(res)
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', metavar="FILE", help='config file path')
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()
    group = GroupBunch(args.config_file, args.test)
    group.forward()

