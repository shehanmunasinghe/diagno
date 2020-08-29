
from ts.torch_handler.base_handler import BaseHandler
import os

import numpy as np
import torch
import json

import logging
logger = logging.getLogger(__name__)

class ModelHandler(BaseHandler):
    def preprocess(self, requests):
        req = requests[0]

        text = req.get("data")
        #if text is None:
        #    text = req.get("body")
        #   sentences = text.decode('utf-8')
        #   logger.info("Received text: '%s'", sentences)
        # logger.info(str(requests))
        logger.info(str(req)[0:100]) #len = 988126
        # logger.info(str(req.get("data")) ) #len =4
        b_array = req.get('inputFile')
        # arr = np.frombuffer(b_array, dtype=np.float32)
        # logger.info(arr.shape)
        b_array=b_array.decode()
        data = json.loads(b_array)
        logger.info(type(data))
        logger.info(len(data))

        inputs_arr = np.array(data,dtype=np.float32)
        inputs_tensor = torch.from_numpy(inputs_arr)
        inputs_tensor = inputs_tensor.unsqueeze(0)

        return inputs_tensor #torch.randn((1,12,4096))


    #def postprocess(self, data):
    #    dict={0:'LAD',1:'TAb',2:'AF',3:'STach',4:'iavb'}
    #    label=int(torch.argmax(data[0]))
    #    return dict[label]

