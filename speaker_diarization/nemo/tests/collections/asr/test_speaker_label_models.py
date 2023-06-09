# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import TestCase

import pytest
from omegaconf import DictConfig

from nemo.collections.asr.models import EncDecSpeakerLabelModel


class EncDecSpeechLabelModelTest(TestCase):
    @pytest.mark.unit
    def test_constructor(self):
        preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
        encoder = {
            'cls': 'nemo.collections.asr.modules.ConvASREncoder',
            'params': {
                'feat_in': 64,
                'activation': 'relu',
                'conv_mask': True,
                'jasper': [
                    {
                        'filters': 512,
                        'repeat': 1,
                        'kernel': [1],
                        'stride': [1],
                        'dilation': [1],
                        'dropout': 0.0,
                        'residual': False,
                        'separable': False,
                    }
                ],
            },
        }

        decoder = {
            'cls': 'nemo.collections.asr.modules.SpeakerDecoder',
            'params': {'feat_in': 512, 'num_classes': 2, 'pool_mode': 'xvector', 'emb_sizes': [1024]},
        }

        modelConfig = DictConfig(
            {
                'preprocessor': DictConfig(preprocessor),
                'encoder': DictConfig(encoder),
                'decoder': DictConfig(decoder),
            },
        )
        speaker_model = EncDecSpeakerLabelModel(cfg=modelConfig)
        speaker_model.train()
        # TODO: make proper config and assert correct number of weights

        # Check to/from config_dict:
        confdict = speaker_model.to_config_dict()
        instance2 = EncDecSpeakerLabelModel.from_config_dict(confdict)
        self.assertTrue(isinstance(instance2, EncDecSpeakerLabelModel))

    @pytest.mark.unit
    def test_ecapa_enc_dec(self):
        preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
        encoder = {
            'cls': 'nemo.collections.asr.modules.ECAPAEncoder',
            'params': {
                'feat_in': 80,
                'filters': [4, 4, 4, 4, 3],
                'kernel_sizes': [5, 3, 3, 3, 1],
                'dilations': [1, 1, 1, 1, 1],
                'scale': 2,
            },
        }

        decoder = {
            'cls': 'nemo.collections.asr.modules.SpeakerDecoder',
            'params': {'feat_in': 3, 'num_classes': 2, 'pool_mode': 'attention', 'emb_sizes': 192},
        }

        modelConfig = DictConfig(
            {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder),}
        )
        speaker_model = EncDecSpeakerLabelModel(cfg=modelConfig)
        speaker_model.train()
        # TODO: make proper config and assert correct number of weights

        # Check to/from config_dict:
        confdict = speaker_model.to_config_dict()
        instance2 = EncDecSpeakerLabelModel.from_config_dict(confdict)
        self.assertTrue(isinstance(instance2, EncDecSpeakerLabelModel))

    @pytest.mark.unit
    def test_titanet_enc_dec(self):
        preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
        encoder = {
            'cls': 'nemo.collections.asr.modules.ConvASREncoder',
            'params': {
                'feat_in': 64,
                'activation': 'relu',
                'conv_mask': True,
                'jasper': [
                    {
                        'filters': 256,
                        'repeat': 1,
                        'kernel': [1],
                        'stride': [1],
                        'dilation': [1],
                        'dropout': 0.0,
                        'residual': False,
                        'separable': True,
                        'se': True,
                        'se_context_size': -1,
                    }
                ],
            },
        }

        decoder = {
            'cls': 'nemo.collections.asr.modules.SpeakerDecoder',
            'params': {'feat_in': 256, 'num_classes': 2, 'pool_mode': 'attention', 'emb_sizes': [1024]},
        }

        modelConfig = DictConfig(
            {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder),}
        )
        speaker_model = EncDecSpeakerLabelModel(cfg=modelConfig)
        speaker_model.train()
        # TODO: make proper config and assert correct number of weights

        # Check to/from config_dict:
        confdict = speaker_model.to_config_dict()
        instance2 = EncDecSpeakerLabelModel.from_config_dict(confdict)
        self.assertTrue(isinstance(instance2, EncDecSpeakerLabelModel))
