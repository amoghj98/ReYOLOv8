import torch
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import logging
import sys
import cv2
import pdb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning_config import configs, print_args
# from utils.visual_helpers import normalize, convert_to_color


class Args(object):
    def __init__(self):
        self.dt = 1000
        self.nBins = 5
        self.nSpeeds = 4
        self.inp_downscale = 2
        self.height = 480
        self.width = 640
        self.min_speed = 1
        self.ofpd_noise = 0.0  # Noise level for OFPD
        self.traindir = os.path.join('/scratch/gautschi/joshi157/datasets/', 'processed_v2', str(self.dt)+'us', 'train')
        self.testdir = os.path.join('/scratch/gautschi/joshi157/datasets/', 'processed_v2', str(self.dt)+'us', 'test')
        self.visdir = os.path.join('/scratch/gautschi/joshi157/datasets/', 'processed_v2', str(self.dt)+'us', 'vis')
        self.mulObjdir = os.path.join('/scratch/gautschi/joshi157/datasets/', 'processed_v2', str(self.dt)+'us', 'mulObj')
        self.groupname = 'objects'
        self.data_reduction_train = {10000: [11, 3.7, 1.5, 1],      #4100 bins/speed, data_reduction=1
                                 5000: [9.5, 3.2, 1.4, 1],      #10000 bins/speed, data_reduction=1
                                 2000: [12, 4.4, 2, 1.3],       #10000 bins/speed, data_reduction=2
                                 1000: [10, 4, 1.9, 1.2],       #10000 bins/speed, data_reduction=5
                                 500: [10, 4, 2.2, 1.3],        #20000 bins/speed, data_reduction=5
                                 200: [18, 6.8, 4.5, 2.3],      #30000 bins/speed, data_reduction=5
                                 100: [26, 11.7, 7.2, 3.7]}     #40000 bins/speed, data_reduction=5
    
        self.data_reduction_test =  {10000: [20, 4.8, 1.3, 2.6],      #1000 bins/speed, data_reduction=1
                                 5000: [15, 4, 1.5, 2.5],       #2000 bins/speed, data_reduction=1
                                 2000: [18, 5, 2.5, 3.5],       #2000 bins/speed, data_reduction=2    
                                 1000: [8, 3, 1, 1.5],          #2000 bins/speed, data_reduction=5
                                 500: [10, 3, 1.5, 2],          #3500 bins/speed, data_reduction=5
                                 200: [18, 8, 4, 4],            #4500 bins/speed, data_reduction=5
                                 100: [21, 10.2, 5.6, 4.8]}     #8000 bins/speed, data_reduction=5


class MOTE_Dataset_Parallel(Dataset):
    def __init__(self, mode='train'):
        # args = {}
        args = Args()
        self.args = args
        self.dt = args.dt
        self.nBins = args.nBins
        self.nSpeeds = args.nSpeeds
        self.groupname = args.groupname
        self.scaling_factor = args.inp_downscale
        # self.data_reduction = args.data_reduction

        self.height = args.height #480 x| Vertical
        self.width = args.width #640 y-- Horizontal

        if mode == 'train':
            self.datadir = args.traindir
            self.data_reduction = args.data_reduction_train
            logging.info(f'\nTraining DataSet...')
        elif mode == 'test':
            self.datadir = args.testdir
            self.data_reduction = args.data_reduction_test
            logging.info(f'\nTesting DataSet...')

        self.sf = {}
        self.sg = {}
        self.length = 0
        self.speedbin_lens = []
        self.speedbin_event_lens = []
        self.invalid = 0
        for sidx in range(self.nSpeeds):
            self.sf[sidx] = h5py.File(os.path.join(self.datadir, 'speed_bin' + str(sidx+args.min_speed) +'.hdf5'), "r")
            self.sg[sidx] = self.sf[sidx][self.groupname]
            self.speedbin_lens.append(int(self.sg[sidx]['pos_txed'].shape[0]//self.data_reduction[self.dt][sidx+args.min_speed-1]))
            corr_index = self.sg[sidx]['speed_corr_map'][self.speedbin_lens[sidx]].item()
            # pdb.set_trace()
            self.speedbin_event_lens.append(self.sg[sidx]['events'][:corr_index].shape[0])
            self.length += self.speedbin_lens[sidx]

        self.speedbin_cumsum = np.cumsum(np.array(self.speedbin_lens))
        
        logging.info(f'Dataset Keys: {self.sg[0].keys()}')
        logging.info(f'Speed Bin Lengths: {self.speedbin_lens}')
        logging.info(f'Speed Bin Cumsum: {self.speedbin_cumsum}')
        logging.info(f'Speed Bin Event Lengths: {self.speedbin_event_lens}')
        logging.info(f'Total Dataset Length: {self.length}')

        self.events = {}
        self.pos_gt = {}
        self.speed_gt = {}
        self.dir_gt = {}
        self.corr_map = {}

        logging.info(f'Reading HDF5 data into RAM! Please be patient...')

        for idx in range(self.nSpeeds):
            self.events[idx] = self.sg[idx]['events'][:self.speedbin_event_lens[idx]]
            self.pos_gt[idx] = self.sg[idx]['pos_txed'][:self.speedbin_lens[idx]]
            self.speed_gt[idx] = self.sg[idx]['mod_vel_mps'][:self.speedbin_lens[idx]]
            self.dir_gt[idx] = self.sg[idx]['vel_hding'][:self.speedbin_lens[idx]]
            self.corr_map[idx] = self.sg[idx]['speed_corr_map'][:self.speedbin_lens[idx]+1]

        for sidx in range(self.nSpeeds):
            self.sf[sidx].close()
        
        logging.info('[DONE]')
    
    def addNoise(self, eventBins, noise_level=0.1):
        # eventBins shape: (480, 640, nBins) / self.scaling_factor
        nBins = eventBins.shape[2]
        noisy_events = eventBins.copy()
        numNoisyPix = np.ceil(noise_level*eventBins.shape[0]*eventBins.shape[1])
        y = np.random.randint(0, eventBins.shape[0]-1, int(numNoisyPix))[:, np.newaxis]
        x = np.random.randint(0, eventBins.shape[1]-1, int(numNoisyPix))[:, np.newaxis]
        for bin in range(nBins):
            noisy_events[y, x, bin] = eventBins[:, :, bin].max()
        return noisy_events

    def get_speed_bin(self, index):
        split = index//self.speedbin_cumsum

        if split.any() == False: #all zeros => 1st speed bin
            hdf5_idx = 0
            inp_idx = index
        else:
            hdf5_idx = split.nonzero()[0][-1] + 1 #last (highest) non-zero element - The last element of split will NEVER be 1 as index is always lower than it
            inp_idx = index - self.speedbin_cumsum[hdf5_idx-1]

        return hdf5_idx, inp_idx

    def get_event_bins(self, events, valid_pose=True):
        event_bins = np.zeros((self.height//self.scaling_factor, self.width//self.scaling_factor, self.nBins), dtype=float) #Currently merging polarities
        
        if valid_pose:
            total_events = events.shape[0]

            for bin in range(self.nBins):
                bin_start = int(bin*total_events//self.nBins)
                bin_end = int((bin+1)*total_events//self.nBins)
                evx = (events[bin_start:bin_end, 0]//self.scaling_factor).astype(int)
                evy = (events[bin_start:bin_end, 1]//self.scaling_factor).astype(int)

                event_bins[evy, evx, bin] += 1
        else:
            self.invalid += 1

        return event_bins
    
    def get_valid_pose(self, pos_txed):
        xlimit = 25
        ylimit = 25

        valid_pose = True

        if ((pos_txed[0]-xlimit) < 0) or ((pos_txed[0]+xlimit) > self.width//self.scaling_factor):
            valid_pose = False

        if ((pos_txed[1]-ylimit) < 0) or ((pos_txed[1]+ylimit) > self.height//self.scaling_factor):
            valid_pose = False

        return valid_pose


    def __getitem__(self, index):

        hdf5_idx, inp_idx = self.get_speed_bin(index)

        gt_pos_txed = self.pos_gt[hdf5_idx][inp_idx]//self.scaling_factor
        gt_pos_txed[0] = np.clip(gt_pos_txed[0], 0, self.width//self.scaling_factor-1)
        gt_pos_txed[1] = np.clip(gt_pos_txed[1], 0, self.height//self.scaling_factor-1)

        gt_mod_vel_mps = self.speed_gt[hdf5_idx][inp_idx]
        gt_vel_hding = self.dir_gt[hdf5_idx][inp_idx]
        gt_vel_hding_degrees = 180*self.dir_gt[hdf5_idx][inp_idx]/np.pi
        gt_vel_hding_degrees[gt_vel_hding_degrees<0] += 360.0

        gt_dir_x = np.cos(gt_vel_hding)
        gt_dir_y = -np.sin(gt_vel_hding)
        gt_dir = np.concatenate((gt_dir_x, gt_dir_y), axis=0)[:, np.newaxis]

        evf = self.corr_map[hdf5_idx][inp_idx]
        evl = self.corr_map[hdf5_idx][inp_idx+1]
        events = self.events[hdf5_idx][int(evf):int(evl)]

        valid_pose = self.get_valid_pose(gt_pos_txed)
        # valid_pose=True

        event_bins = self.get_event_bins(events, valid_pose)
        if self.args.ofpd_noise !=0:
            event_bins = self.addNoise(event_bins, noise_level=self.args.ofpd_noise)

        # if valid_pose:
        #     return torch.from_numpy(event_bins), torch.from_numpy(gt_pos_txed), torch.from_numpy(gt_dir), torch.from_numpy(gt_mod_vel_mps), torch.from_numpy(gt_vel_hding_degrees)#, self.invalid
        # else:
        #     return torch.from_numpy(event_bins), torch.from_numpy(np.zeros_like(gt_pos_txed)), torch.from_numpy(np.zeros_like(gt_dir)), torch.from_numpy(np.zeros_like(gt_mod_vel_mps)), torch.from_numpy(np.zeros_like(gt_vel_hding_degrees))

        ret_dict = {}

        evt_repr = torch.permute(torch.from_numpy(event_bins), (2, 0, 1)).unsqueeze(0).repeat(11, 1, 1, 1)  # (1, nBins, height, width)
        # print(f'evt_repr shape: {evt_repr.shape}')
        # ip = evt_repr[:, 0, :, :].repeat(2, 1, 1)
        # for i in range(self.nBins):
        #     ip = torch.cat((ip, evt_repr[:, i, :, :].repeat(2, 1, 1)), dim=0)
        # ip = torch.cat((ip, evt_repr[:, self.nBins - 1, :, :]), dim=0).unsqueeze(0)
        ret_dict['img'] = evt_repr
        ret_dict['cls'] = torch.ones(1).unsqueeze(0)#.unsqueeze(0).unsqueeze(0) # torch.from_numpy(np.concatenate(classes, axis = 0, dtype = np.float64, casting = "unsafe")) (1, 1)
        # ret_dict['sequence'] =  sequence_data
        # ret_dict['vid_file'] = img_file
        ret_dict['vid_pos'] = torch.from_numpy(np.array(index))
        ret_dict['clip_pos'] = torch.zeros((11,))# torch.from_numpy(np.array(frame))
        ret_dict['batch_idx'] = torch.zeros(1).unsqueeze(0)#.unsqueeze(0).unsqueeze(0) # (1, 1)
        #
        if valid_pose:
            # ret_dict['img']  = torch.permute(torch.from_numpy(event_bins), (2, 0, 1)).unsqueeze(0).repeat(11, 1, 1, 1) #torch.from_numpy(event_bins) (nBins, height, width)
            # print(f'img shape: {ret_dict["img"].shape}')
            ret_dict['bboxes'] = torch.cat((torch.from_numpy(gt_pos_txed), torch.from_numpy(gt_dir)), dim=0).squeeze()  # 50 is a dummy value for width and height (1, 4)
            # print(f'bboxes shape: {ret_dict["bboxes"].shape}')
        else:
            # ret_dict['img'] = torch.cat((evt_repr[:, i, :, :].repeat(1, 2, 1, 1) for i in range(self.nBins)), dim=0)
            # ret_dict['img'] = torch.cat((ret_dict['img'], evt_repr[:, self.nBins - 1, :, :]), dim=0)
            # ret_dict['img']  = torch.permute(torch.from_numpy(event_bins), (2, 0, 1)).unsqueeze(0).repeat(11, 1, 1, 1) #torch.from_numpy(event_bins) (nBins, height, width)
            ret_dict['bboxes'] = torch.cat((torch.from_numpy(np.zeros_like(gt_pos_txed)), torch.zeros_like(torch.from_numpy(gt_pos_txed))), dim=0).squeeze()  # 50 is a dummy value for width and height (1, 4)
            # ret_dict['sequence'] =  sequence_data
            # ret_dict['vid_file'] = img_file

        return ret_dict

    def __len__(self):
        return self.length