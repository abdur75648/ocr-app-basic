import torch.nn as nn

from .modules.transformation import TPS_SpatialTransformerNetwork
from .modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor, UNet_FeatureExtractor
from .modules.sequence_modeling import BidirectionalLSTM
from .modules.prediction import Attention
from .modules.dropout_layer2 import dropout_layer


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        #else:
            #print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'UNet':
            self.FeatureExtraction = UNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        
        if opt.SequenceModeling == 'BiLSTM':
            
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            
            self.SequenceModeling_output = opt.hidden_size
        
        
        elif opt.SequenceModeling == 'DBiLSTM':            
            self.dropout1 = nn.Sequential(
                dropout_layer()
            )
            self.dropout2 = nn.Sequential(
                dropout_layer()
            )
            self.dropout3 = nn.Sequential(
                dropout_layer()
            )
            self.dropout4 = nn.Sequential(
                dropout_layer()
            )
            self.dropout5 = nn.Sequential(
                dropout_layer()
            )
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
                    
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        
        elif self.stages['Seq'] == 'DBiLSTM':
            # contextual_feature = self.SequenceModeling(visual_feature)
            visual_feature_after_dropout1 = self.dropout1(visual_feature)
            visual_feature_after_dropout2 = self.dropout2(visual_feature)
            visual_feature_after_dropout3 = self.dropout3(visual_feature)
            visual_feature_after_dropout4 = self.dropout4(visual_feature)
            visual_feature_after_dropout5 = self.dropout5(visual_feature)
            contextual_feature1 = self.SequenceModeling(visual_feature_after_dropout1)
            contextual_feature2 = self.SequenceModeling(visual_feature_after_dropout2)
            contextual_feature3 = self.SequenceModeling(visual_feature_after_dropout3)
            contextual_feature4 = self.SequenceModeling(visual_feature_after_dropout4)
            contextual_feature5 = self.SequenceModeling(visual_feature_after_dropout5)
            contextual_feature =  ( (contextual_feature1).add ((contextual_feature2).add(((contextual_feature3).add(((contextual_feature4).add(contextual_feature5)))))) ) * (1/5)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction
