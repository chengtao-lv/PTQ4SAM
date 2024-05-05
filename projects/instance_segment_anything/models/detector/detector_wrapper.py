import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule

from mmcv import Config
from mmdet.apis import inference_detector, init_detector

class DetectorWrapper(BaseModule):
    def __init__(self,
                 args=None,
                 init_cfg=None):
        super(DetectorWrapper, self).__init__(init_cfg)
        
        config = Config.fromfile(args.det_config)
        self.model = init_detector(config, args.det_weight, device='cpu', cfg_options={}).cuda()

        self.cls_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                          55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                          82, 84, 85, 86, 87, 88, 89, 90]
        

    def forward(self,
                img,
                img_metas):
        """Forward function for training mode.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        """
        
        return inference_detector(self.model, img_metas[0]['filename'],)
        

    def simple_test(self, img, img_metas, rescale=False):
        # out: dict
        out = self(img, img_metas)
        # if rescale:
        #     ori_target_sizes = [meta_info['ori_shape'][:2] for meta_info in img_metas]
        # else:
        #     ori_target_sizes = [meta_info['img_shape'][:2] for meta_info in img_metas]
        # ori_target_sizes = (out['pred_logits']).new_tensor(ori_target_sizes, dtype=torch.int64)
        # # results: List[dict(scores, labels, boxes)]
        # results = self.box_postprocessor(out, ori_target_sizes)
        import numpy as np
        boxes = np.ndarray((0,4))
        scores = np.ndarray((0))
        labels = []
        results = {}
        
        for cls in range(len(out)):
            cls_pred = out[cls]
            if len(cls_pred) == 0: 
                continue
            labels.extend([cls for _ in range(len(cls_pred))])
            boxes = np.concatenate([boxes, cls_pred[:,:4]],axis=0)
            scores = np.concatenate([scores, cls_pred[:,4]],axis=0)
        results['labels'] = torch.tensor(labels).cuda()
        results['scores'] = torch.from_numpy(scores).cuda()
        results['boxes'] = torch.from_numpy(boxes).cuda()
        
        if len(results['labels']) == 0:
            results['boxes'] = torch.zeros((1,4)).cuda()
            results['scores'] = torch.tensor([0.1,]).cuda()
            results['labels'] = torch.tensor([0,]).cuda()
        return [results,]
