import numpy as np





from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr.transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
import seqloss as seqloss
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from gumbel_vector_quantizer import GumbelVectorQuantizer
import torch.nn.functional as F
from torch import batch_norm_gather_stats, nn
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union
from espnet2.train.preprocessor import CommonPreprocessor
from espnet.nets.pytorch_backend.nets_utils import pad_list
from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
    LabelSmoothingLoss2
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr.transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from gumbel_vector_quantizer import GumbelVectorQuantizer
import torch.nn.functional as F
from torch import nn
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder


import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter



def read_2column_text(path: Union[Path, str]) -> Dict[str, str]:
    """Read a text file having 2 column as dict object.

    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> read_2column_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """
    assert check_argument_types()

    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = v
    return data
token_list=[]
with open("data/en_token_list/bpe_unigram5000/tokens.txt") as f:
    for line in f:
        line=line.strip()
        token_list.append(line)

process=CommonPreprocessor(
    train=True,
    token_type='bpe',
    token_list=token_list,
    bpemodel='data/en_token_list/bpe_unigram5000/bpe.model',
    non_linguistic_symbols=None,
    text_cleaner=None,
    g2p_type=None,
    # NOTE(kamo): Check attribute existence for backward compatibility
    rir_scp=None,
    rir_apply_prob=None,
    text_name = "text",
    text_phone_name = "text_phone",

)

class EncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module=None
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
            self.feed_forward(x)
        )
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask

















class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        self.line = nn.Linear(256, 81) 
        self.embeddings = torch.nn.Embedding(
            num_embeddings=81,  # the last for PAD valu衰减直。这里的大小设置的是整个词典的大小。
            embedding_dim=256)
        #self.line_emb=torch.nn.Linear(256, 256)
        #self.norm1 = nn.LayerNorm(256, eps=1e-5)
        
        #self.gum_linear1= torch.nn.Linear(256, 256 )
        #self.gum_linear2 = torch.nn.Linear(256, 81 )

        self.line_phone=torch.nn.Linear(256, 256)



        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args=(4, 256, 0.1, False)

        activation = get_activation("swish")
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args=(256, 1024, 0.1, activation)

        encoder_selfattn_layer = RelPositionMultiHeadedAttention


        convolution_layer = ConvolutionModule
        convolution_layer_args = (256, 31, activation)

        macaron_style=True
        use_cnn_module=True
        self.change_embedding1=EncoderLayer(
                256,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                0.1,
                True,
                False,
                0.0,
            )
        self.change_embedding2=EncoderLayer(
                256,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                0.1,
                True,
                False,
                0.0,
            )
        self.change_embedding3=EncoderLayer(
                256,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                0.1,
                True,
                False,
                0.0,
            )
        self.change_embedding4=EncoderLayer(
                256,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                0.1,
                True,
                False,
                0.0,
            )
        self.change_embedding5=EncoderLayer(
                256,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                0.1,
                True,
                False,
                0.0,
            )
        self.change_embedding6=EncoderLayer(
                256,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                0.1,
                True,
                False,
                0.0,
            )
        self.dropout = torch.nn.Dropout(p=0.02)

        self.embed =RelPositionalEncoding(256,0.1)
        self.after_norm1 = torch.nn.LayerNorm(256)


        self.ctc=CTC(81, 256)
        self.l2_loss = seqloss.MaskedMSELoss()
   
        self.criterion_att = LabelSmoothingLoss(
            size=81,
            padding_idx=-1,
            smoothing=0.1,
            normalize_length=False,
        )
        self.criterion_att2 = LabelSmoothingLoss2(
            size=81,
            padding_idx=-1,
            smoothing=0.1,
            normalize_length=False,
        )
    def forward(self, tensor_phone,lens_phone,tensor_text,lens_text,tensor_phone_right):
        batch_size=tensor_phone.shape[0]

        device="cuda"
        phone_embedding= self.embeddings.weight.unsqueeze(0).repeat(batch_size,1,1).transpose(1,2).to(device)


        # phone_len=tensor_phone.shape[1]
        
        # #B:T:81
        # scores = 0.0000000001*torch.ones(batch_size,phone_len,81).to(device)
        # for B in range(tensor_phone.shape[0]):
        #     for index in range(tensor_phone[B].shape[0]):
        #         phone_name=tensor_phone[B][index]
        #         if phone_name!=-1:
        #             scores[B][index][phone_name]=0.9999

        encoder_out_lens=lens_phone
        encoder_mask=tensor_phone_right.ge(0).unsqueeze(1)

        scores=torch.nn.functional.one_hot(tensor_phone, num_classes=81).to(torch.float32)
        #p_attn = self.dropout(scores)
        p_attn=scores




        cross_attention_out = torch.matmul(p_attn, phone_embedding.transpose(1,2))

        cross_attention_out,pos_emb=self.embed(cross_attention_out)

        final_encoder_out, chunk_masks  = self.change_embedding1((cross_attention_out,  pos_emb),  encoder_mask)
        final_encoder_out, chunk_masks  = self.change_embedding2(final_encoder_out,  encoder_mask)
        final_encoder_out, chunk_masks  = self.change_embedding3(final_encoder_out,  encoder_mask)
        final_encoder_out, chunk_masks  = self.change_embedding4(final_encoder_out,  encoder_mask)
        final_encoder_out, chunk_masks  = self.change_embedding5(final_encoder_out,  encoder_mask)
        final_encoder_out, chunk_masks  = self.change_embedding6(final_encoder_out,  encoder_mask)


        y=self.after_norm1(final_encoder_out[0])

        #one_hot

        # target_onehot=torch.eye(81).to(·torch.float32).to("cuda")
        # one_in=torch.matmul(target_onehot.detach(), phone_embedding.transpose(1,2))
        # one_in,pos_emb=self.embed(one_in)
        # one_in=      F.relu( self.line_phone(one_in))

        # one_in, chunk_masks  = self.change_embedding1((one_in,  pos_emb),  encoder_mask)
        # one_in=self.after_norm1(one_in[0])
        # one_in, chunk_masks  = self.change_embedding2((one_in,  pos_emb),  encoder_mask)
        # one_in=self.line(self.after_norm2(one_in[0])).softmax(-1)
        # one_in_Loss=torch.nn.MSELoss(target_onehot,one_in)
    
        target_onehot=torch.nn.functional.one_hot((~tensor_phone_right.ge(0)+tensor_phone_right), num_classes=81).to(torch.float32)
        mask2=(~torch.eq(tensor_phone_right,tensor_phone).unsqueeze(1))*encoder_mask
        (~tensor_phone_right.ge(0)+tensor_phone_right)
        loss1=self.criterion_att(self.ctc.ctc_lo(y),tensor_phone_right)
        #ML=loss

       # loss2=self.criterion_att2(self.ctc.ctc_lo(y),tensor_phone_right,mask2)

        #one_in_Loss=self.l2_loss(target_onehot,self.ctc.ctc_lo(y).softmax(-1),  mask=mask2.transpose(1,2)) #+ 0.1* self.l2_loss(target_onehot,self.ctc.ctc_lo(y).softmax(-1),  mask=encoder_mask.transpose(1,2)) + 0.1*self.ctc(y, lens_phone, tensor_text, lens_text)
        # one_in_Loss=0.7*self.l2_loss(target_onehot,self.ctc.ctc_lo(y).softmax(-1),  mask=mask2.transpose(1,2)) + 0.1* self.l2_loss(target_onehot,self.ctc.ctc_lo(y).softmax(-1),  mask=encoder_mask.transpose(1,2)) + 0.1*self.ctc(y, lens_phone, tensor_text, lens_text)
        # #one_in_Loss=torch.nn.functional.cross_entropy(target_onehot,self.ctc.ctc_lo(y).softmax(-1))

        # #ML=0.2*self.ctc(y, lens_phone, tensor_text, lens_text) + 0.8* one_in_Loss
        ML= loss1

        one_in_Loss=None
        return ML, encoder_mask,one_in_Loss



model = my_model()

model=model.to("cuda")

#加载ctc模型参数
# checkpoint = torch.load("/home3/yuhang001/espnet/egs2/librispeech_100/letter/exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic/best_letter_10.pth",map_location='cuda')
# model.line.weight=Parameter(checkpoint["ctc_phone.ctc_lo.weight"])
# model.line.bias=Parameter(checkpoint["ctc_phone.ctc_lo.bias"])
# model.ctc.ctc_lo.weight=Parameter(checkpoint["ctc_phone.ctc_lo.weight"])
# model.ctc.ctc_lo.bias=Parameter(checkpoint["ctc_phone.ctc_lo.bias"])
# # #加载训好模型的参数
checkpoint = torch.load("dict_model/0params_with_transformer8.pth")
model.load_state_dict(checkpoint)




model.train()



# for para in model.line.parameters():
#     para.requires_grad = False

# for para in model.ctc.parameters():
#     para.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)





#读取，文本
#设置batch
#转换成id

#转换成tensor
loader_dict = {}
loader_text = read_2column_text("dump/raw/train_960_text_debug/text" )
loader_phone = read_2column_text("dump/raw/train_960_text_debug/phone_text_with_id" )

batch_size=300
batch_items=[]
index=0

sort_utt_dict=[]
with open("dump/raw/train_960_text_debug/sort_utt3") as f:

    for line in f:
        line=line.strip()
        sort_utt_dict.append(line)


for i in sort_utt_dict:
    if index==0:
        temp={}
    tempp=process._phone_process({"text_phone":loader_phone[i]},i)
    temp[i]=(process._phone_process_old({"text_phone":loader_phone[i]},i)['text_phone'],  tempp['text_phone'],tempp['phone_right'])
    if index%batch_size==0 and index!=0:
        

        batch_items.append(temp)
        temp={}
    index+=1


epoch = 20
l2_loss = seqloss.MaskedMSELoss()
#l2_loss = seqloss.MaskedL1Loss()

with open("./log33","w") as f:
    for i in range(epoch):
    #整个成tensor ,进行padding和mask
        for batch in batch_items:
            text=[]
            phone=[]
            phone_right=[]
            for k ,v in batch.items():
                text.append(v[0])
                phone.append(v[1])
                phone_right.append(v[2])
                #if len(v[1])!=len(v[2]):
                #    print(1)

            tensor_list_text = [torch.from_numpy(a) for a in text]

            tensor_list_phone = [torch.from_numpy(a) for a in phone]
            tensor_list_phone_right = [torch.from_numpy(a) for a in phone_right]

            tensor_text = pad_list(tensor_list_text, 0)
            tensor_phone = pad_list(tensor_list_phone, 0)
            tensor_phone_right = pad_list(tensor_list_phone_right, -1)

            lens_text = torch.tensor([len(d) for d in text], dtype=torch.long)
            lens_phone = torch.tensor([len(d) for d in phone], dtype=torch.long)



            device= "cuda"

            tensor_phone=tensor_phone.to(device)
            tensor_phone_right=tensor_phone_right.to(device)
            lens_phone=lens_phone.to(device)

            ML,mask,one_in_Loss=model(tensor_phone,lens_phone,tensor_text,lens_text,tensor_phone_right)

            

            



            #ML=ML+one_in_Loss
            optimizer.zero_grad()
            ML.backward()
            optimizer.step()
            
            print(str(ML.clone().detach().cpu().item())+"#################################################\n")
            f.writelines(str(ML.clone().detach().cpu().item())+"##############################################\n")
   
        #torch.save(model.state_dict(), "./dict_model/"+str(i)+'params_with_transformer8.pth')


print(1)
#模型保存参数：torch.save(model.state_dict(), './params.pth')



# X_inv = np.linalg.pinv(my_ny)
# embedding=np.dot(X_inv,target_onehot)

# np.save('embedding.npy',embedding)

# # C = np.linalg.lstsq(my_ny,target_onehot, rcond=None)[0]
# # print()
# # my_ny=np.array([[1,2],[3,4]])
# # target_onehot=np.array([1,2])

# #embeding=np.linalg.solve(my_ny,target_onehot)
# np.save('my_ny.npy',my_ny)

# 4233 256 ; 256 4233 ; 4233 4233
