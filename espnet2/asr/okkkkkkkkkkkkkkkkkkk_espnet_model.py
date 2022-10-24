from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np
from wenet.transformer.vae_decode import vae_decode
import torch
from typeguard import check_argument_types
from style_loss import StyleLoss
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
from torch import nn
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
    MultiHeadedAttention_no_qkv,
    MultiHeadedAttention_no_qkv_gum,
    MultiHeadedAttention_no_qkv_cosin,
    MultiHeadedAttention_no_qkv_cosin2
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


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class EncoderLayer_add(nn.Module):
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
        super(EncoderLayer_add, self).__init__()
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






class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
    ):
    
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )
            self.criterion_phone = LabelSmoothingLoss(
                size=81,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
            self.ctc_phone =  CTC(81, 256)
            self.ctc_phone_other=  CTC(81, 256)
        self.phone_embedding1 = torch.nn.Embedding(
            num_embeddings=81,  # the last for PAD valu衰减直。这里的大小设置的是整个词典的大小。
            embedding_dim=256,
          )
        self.phone_embedding2 = torch.nn.Embedding(
            num_embeddings=81,  # the last for PAD valu衰减直。这里的大小设置的是整个词典的大小。
            embedding_dim=256,
          )

        #encoder出来本来就是after_norm,这是给对齐网络用的
        self.after_norm = torch.nn.LayerNorm(256)
        self.after_norm_right = torch.nn.LayerNorm(256)
        #self.after_norm_text=torch.nn.LayerNorm(256)
     


        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats


        # encoder_selfattn_layer = RelPositionMultiHeadedAttention
        # encoder_selfattn_layer_args=(4, 256, 0.1, False)

        # activation = get_activation("swish")
        # positionwise_layer = PositionwiseFeedForward
        # positionwise_layer_args=(256, 1024, 0.1, activation)

        # encoder_selfattn_layer = RelPositionMultiHeadedAttention


        # convolution_layer = ConvolutionModule
        # convolution_layer_args = (256, 31, activation)

        # macaron_style=True
        # use_cnn_module=True
        # self.change_embedding1=EncoderLayer_add(
        #         256,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args),
        #         positionwise_layer(*positionwise_layer_args),
        #         positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        #         convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        #         0.1,
        #         True,
        #         False,
        #         0.0,
        #     )
        # self.change_embedding2=EncoderLayer_add(
        #         256,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args),
        #         positionwise_layer(*positionwise_layer_args),
        #         positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        #         convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        #         0.1,
        #         True,
        #         False,
        #         0.0,
        #     )
        # self.change_embedding3=EncoderLayer_add(
        #         256,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args),
        #         positionwise_layer(*positionwise_layer_args),
        #         positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        #         convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        #         0.1,
        #         True,
        #         False,
        #         0.0,
        #     )
        # self.change_embedding4=EncoderLayer_add(
        #         256,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args),
        #         positionwise_layer(*positionwise_layer_args),
        #         positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        #         convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        #         0.1,
        #         True,
        #         False,
        #         0.0,
        #     )
        # self.change_embedding5=EncoderLayer_add(
        #         256,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args),
        #         positionwise_layer(*positionwise_layer_args),
        #         positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        #         convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        #         0.1,
        #         True,
        #         False,
        #         0.0,
        #     )

        # self.change_embedding6=EncoderLayer_add(
        #         256,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args),
        #         positionwise_layer(*positionwise_layer_args),
        #         positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        #         convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        #         0.1,
        #         True,
        #         False,
        #         0.0,
        #     )
        # self.norm6=torch.nn.LayerNorm(256)

        


        # self.change_embedding11=EncoderLayer_add(
        #         256,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args),
        #         positionwise_layer(*positionwise_layer_args),
        #         positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        #         convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        #         0.1,
        #         True,
        #         False,
        #         0.0,
        #     )
        # self.change_embedding22=EncoderLayer_add(
        #         256,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args),
        #         positionwise_layer(*positionwise_layer_args),
        #         positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        #         convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        #         0.1,
        #         True,
        #         False,
        #         0.0,
        #     )
        # self.change_embedding33=EncoderLayer_add(
        #         256,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args),
        #         positionwise_layer(*positionwise_layer_args),
        #         positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        #         convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        #         0.1,
        #         True,
        #         False,
        #         0.0,
        #     )
        # self.norm33=torch.nn.LayerNorm(256)
        # self.change_embedding44=EncoderLayer_add(
        #         256,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args),
        #         positionwise_layer(*positionwise_layer_args),
        #         positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        #         convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        #         0.1,
        #         True,
        #         False,
        #         0.0,
        #     )
        # self.change_embedding55=EncoderLayer_add(
        #         256,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args),
        #         positionwise_layer(*positionwise_layer_args),
        #         positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        #         convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        #         0.1,
        #         True,
        #         False,
        #         0.0,
        #     )
        # self.change_embedding66=EncoderLayer_add(
        #         256,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args),
        #         positionwise_layer(*positionwise_layer_args),
        #         positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        #         convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        #         0.1,
        #         True,
        #         False,
        #         0.0,
        #     )

        self.decoder_out =torch.nn.Linear(            256,29   )
        self.decoder_out2 =torch.nn.Linear(            256, 29   )
        activation = get_activation("swish")

        drop_rate=0.0
        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args=(4, 256, drop_rate, False)

 
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args=(256, 1024, drop_rate, activation)

        encoder_selfattn_layer = RelPositionMultiHeadedAttention


        self.encoders_add= repeat(
            3,
            lambda lnum: EncoderLayer(
                256,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                None,
                None,
                drop_rate,
                True,
                False,
                0.0,
            ),
        )



        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.embed =RelPositionalEncoding(256,drop_rate)
        self.self_attn=  MultiHeadedAttention_no_qkv(
                    4, 256, 0
                )
        self.self_attn2=  MultiHeadedAttention_no_qkv_cosin(
                    4, 256, 0
                )
        self.self_attn3=  MultiHeadedAttention_no_qkv_cosin2(
                    4, 256, 0
                )

        self.l2_loss = seqloss.MaskedMSELoss()
        self.l2_loss2 = seqloss.MaskedMSELoss2()
        self.embed =RelPositionalEncoding(256,0.1)
        self.style_loss=StyleLoss()



    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_phone: torch.Tensor,
        phone_right:torch.Tensor,
        text_phone_lengths: torch.Tensor,
        phone_right_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        #只有文本
        # if len(speech[0])==5 and speech[0][0]==1.0:
        if False:
            train_text_encoder=True
            if train_text_encoder:
                # for data-parallel
                text = text[:, : text_lengths.max()]
                text_phone=text_phone[:, : text_phone_lengths.max()]

                batch_size=speech.shape[0]
                
                phone_embedding1= self.phone_embedding1.weight.unsqueeze(0).repeat(batch_size,1,1) 



                device=text.device
                phone_len=text_phone.shape[1]
                
    

                text_phone_in_pad,text_phone_out_pad = add_sos_eos(text_phone, 0, 0, 0)
                text_phone_in_pad_flag=text_phone_in_pad
                text_phone_mask=text_phone_in_pad.ge(0).unsqueeze(1)
                text_phone_in_pad=text_phone_in_pad+(~text_phone_in_pad.ge(0))
                text_phonein_lens = text_phone_lengths + 1
                #text_phone 去除-1



                # text_phone_in_pad_masked=text_phone_in_pad.clone().detach()
                # # text_phone_mask2=text_phone_mask
                # n = text_phone_in_pad_masked.numel()
                # m = int(round(n*0.3))
                # indices = np.random.choice(n, m, replace=False) # alternative: indices = torch.randperm(n)[:m]
                # text_phone_in_pad_masked.flatten()[indices] = 0

                # scores2_mask=torch.nn.functional.one_hot(text_phone_in_pad, num_classes=81).to(torch.float32)
                # text_phone_embedding_mask=torch.matmul(scores2_mask ,phone_embedding1 )
                # text_phone_embedding_mask,pos_emb=self.embed(text_phone_embedding_mask)






                # #encoder，没有mask的text_encoder输出
                # text_phone_embedding_mask, text_phone_mask = self.encoders_add((text_phone_embedding_mask,pos_emb), text_phone_mask)
                # text_phone_embedding_mask=text_phone_embedding_mask[0]

                # loss_other_ctc3 = self.criterion_phone( self.ctc_phone.ctc_lo(  text_phone_embedding_mask ) , text_phone_in_pad_flag)

               

                ###########################################################
                #Phone 乘上embedding
                scores2=torch.nn.functional.one_hot(text_phone_in_pad, num_classes=81).to(torch.float32)
                text_phone_embedding=torch.matmul(scores2 ,phone_embedding1 )
                text_phone_embedding,pos_emb=self.embed(text_phone_embedding)

                #encoder，没有mask的text_encoder输出
                text_phone_embedding, text_phone_mask = self.encoders_add((text_phone_embedding,pos_emb), text_phone_mask)
                text_phone_embedding=text_phone_embedding[0].detach()



                #插入blank,增加错误。text_phone_embedding B* leng* 256  .unsqueeze(-1).repeat(1,1,2).reshape(2,2,2).reshape(2,-1)
                double_embding=text_phone_embedding.reshape(-1,256).unsqueeze(-1).repeat(1,1,2).reshape(-1,256,2).reshape(batch_size,-1,256)
                # n = text_phone.numel()
                # m = int(round(n*0.4))
                # indices = np.random.choice(n, m, replace=False) # alternative: indices = torch.randperm(n)[:m]
                # # text_phone_in_pad=text_phone
                # text_phone_in_pad = text_phone_in_pad.clone().detach()
                # text_phone_in_pad.flatten()[indices] = 0

                # #double_text
                double_text=text_phone_in_pad.unsqueeze(-1).repeat(1,1,2).reshape(batch_size,-1)
                double_text_masked=double_text.clone().detach()
                # text_phone_mask2=text_phone_mask
                n = double_text_masked.numel()
                m = int(round(n*0.3))
                indices = np.random.choice(n, m, replace=False) # alternative: indices = torch.randperm(n)[:m]
                double_text_masked.flatten()[indices] = 0


                mask1=double_text_masked.ge(1).unsqueeze(-1).repeat(1,1,256)
                mask2=~double_text_masked.ge(1).unsqueeze(-1).repeat(1,1,256)
                #zero_phone 
                #不要用zero，而是用随机token生成的embedding序列
                #double_embding随机打乱。
                # zero_phone=self.phone_embedding2.weight[0].unsqueeze(0).unsqueeze(0).repeat(batch_size,text_phone_in_pad.shape[1]*2,1)
                zero_phone=torch.flip(double_embding.clone().detach(),[0])

                text_phone_embedding=(double_embding*mask1+zero_phone*mask2)
                # text_phone_embedding=double_embding
                text_phone_embedding=self.after_norm(text_phone_embedding)





            
                cross_attention_out,pos_emb1=self.embed(text_phone_embedding)
                final_encoder_out=(cross_attention_out,pos_emb1)
                encoder_out, encoder_out_lens ,encoder_mask= self.encode2(final_encoder_out, text_phonein_lens*2)




                #人为构造中间层输出，传导后续的模型当中
                #加一个if判断，
                #xs_pad, masks=self.encoder.encoders[9](encoder_out[1][0][1],encoder_mask)

                intermediate_outs = None
                if isinstance(encoder_out, tuple):
                    intermediate_outs = encoder_out[1]
                    encoder_out = encoder_out[0]
               
              
               
                stats = dict()



                # 1. CTC branch
                if self.ctc_weight != 0.0:
                    loss_ctc, cer_ctc = self._calc_ctc_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )

                    # Collect CTC branch stats
                    stats["loss_ctc2"] = loss_ctc.detach() if loss_ctc is not None else None
                   









                # 2b. Attention decoder branch
                if self.ctc_weight != 1.0:
                    loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )
                    None

                # 3. CTC-Att loss definition
                if self.ctc_weight == 0.0:
                    loss = loss_att
                elif self.ctc_weight == 1.0:
                    loss = loss_ctc
                else:

                    loss =0.3 * loss_ctc + 0.7 * loss_att #+loss_other_ctc3  #+ 0.2* loss_mse_right

                # Collect Attn branch stats
                stats["loss_att2"] = loss_att.detach() if loss_att is not None else None
                #stats["loss_mse_right"] = loss_mse_right.detach() 
                loss_other_ctc3=None
                stats["loss_other_ctc3"] = loss_other_ctc3.detach() if loss_other_ctc3 is not None else None
                stats["acc2"] = acc_att
                stats["cer"] = cer_att
                # Collect total loss stats
                stats["loss"] = loss.detach()
                ##############################################









                

                # Collect Attn branch stats
                

                # force_gatherable: to-device and to-tensor if scalar for DataParallel
                loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
                return loss, stats, weight
            else:





        # for data-parallel
                text = text[:, : text_lengths.max()]
                text_phone=text_phone[:, : text_phone_lengths.max()]

                batch_size=speech.shape[0]
                
                phone_embedding1= self.phone_embedding1.weight.unsqueeze(0).repeat(batch_size,1,1) 



                device=text.device
                phone_len=text_phone.shape[1]
                
                #B:T:81
                # scores = 0.0000000001*torch.ones(batch_size,phone_len,81).to(device)
                # for B in range(text_phone.shape[0]):
                #     for index in range(text_phone[B].shape[0]):
                #         phone_name=text_phone[B][index]
                #         if phone_name!=-1:
                #             scores[B][index][phone_name]=0.9999

                text_phone_in_pad,text_phone_out_pad = add_sos_eos(text_phone, 0, 0, 0)
                text_phone_in_pad_flag=text_phone_in_pad
                text_phone_mask=text_phone_in_pad.ge(0).unsqueeze(1)
                text_phone_in_pad=text_phone_in_pad+(~text_phone_in_pad.ge(0))
                text_phonein_lens = text_phone_lengths + 1
                #text_phone 去除-1

                #Phone 乘上embedding
                scores2=torch.nn.functional.one_hot(text_phone_in_pad, num_classes=81).to(torch.float32)
                text_phone_embedding=torch.matmul(scores2 ,phone_embedding1 )
                text_phone_embedding,pos_emb=self.embed(text_phone_embedding)

                #encoder，没有mask的text_encoder输出
                text_phone_embedding, text_phone_mask = self.encoders_add((text_phone_embedding,pos_emb), text_phone_mask)
                text_phone_embedding=text_phone_embedding[0]



                #插入blank,增加错误。text_phone_embedding B* leng* 256  .unsqueeze(-1).repeat(1,1,2).reshape(2,2,2).reshape(2,-1)
                double_embding=text_phone_embedding.reshape(-1,256).unsqueeze(-1).repeat(1,1,2).reshape(-1,256,2).reshape(batch_size,-1,256)
                n = text_phone.numel()
                m = int(round(n*0.4))
                indices = np.random.choice(n, m, replace=False) # alternative: indices = torch.randperm(n)[:m]
                # text_phone_in_pad=text_phone
                text_phone_in_pad = text_phone_in_pad.clone().detach()
                text_phone_in_pad.flatten()[indices] = 0

                # #double_text
                double_text=text_phone_in_pad.unsqueeze(-1).repeat(1,1,2).reshape(batch_size,-1)
                mask1=double_text.ge(1).unsqueeze(-1).repeat(1,1,256)
                mask2=~double_text.ge(1).unsqueeze(-1).repeat(1,1,256)
                #zero_phone 
                zero_phone=self.phone_embedding2.weight[0].unsqueeze(0).unsqueeze(0).repeat(batch_size,text_phone_in_pad.shape[1]*2,1)


                text_phone_embedding=(double_embding*mask1+zero_phone*mask2).contiguous().detach()
                # text_phone_embedding=double_embding
                text_phone_embedding=self.after_norm(text_phone_embedding)





            
                cross_attention_out,pos_emb=self.embed(text_phone_embedding)
                final_encoder_out=(cross_attention_out,pos_emb)
                encoder_out, encoder_out_lens ,encoder_mask= self.encode2(final_encoder_out, text_phonein_lens*2)




                #人为构造中间层输出，传导后续的模型当中
                #加一个if判断，
                #xs_pad, masks=self.encoder.encoders[9](encoder_out[1][0][1],encoder_mask)

                intermediate_outs = None
                if isinstance(encoder_out, tuple):
                    intermediate_outs = encoder_out[1]
                    encoder_out = encoder_out[0]
                loss_att, acc_att, cer_att, wer_att = None, None, None, None
                loss_ctc, cer_ctc = None, None
                loss_transducer, cer_transducer, wer_transducer = None, None, None
                stats = dict()



                # 1. CTC branch
                if self.ctc_weight != 0.0:
                    loss_ctc, cer_ctc = self._calc_ctc_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )

                    # Collect CTC branch stats
                    stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
                    stats["cer_ctc"] = cer_ctc









                # 2b. Attention decoder branch
                if self.ctc_weight != 1.0:
                    loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )
                    None

                # 3. CTC-Att loss definition
                if self.ctc_weight == 0.0:
                    loss = loss_att
                elif self.ctc_weight == 1.0:
                    loss = loss_ctc
                else:

                    loss =0.3 * loss_ctc + 0.7 * loss_att  #+ 0.2* loss_mse_right

                # Collect Attn branch stats
                stats["loss_att"] = loss_att.detach() if loss_att is not None else None
                #stats["loss_mse_right"] = loss_mse_right.detach() 
                stats["acc"] = acc_att
                stats["cer"] = cer_att
                stats["wer"] = wer_att

                # Collect total loss stats
                stats["loss"] = loss.detach()

                # force_gatherable: to-device and to-tensor if scalar for DataParallel
                loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
                return loss, stats, weight




#音频带文本
##################################################################################################
        else:
            assert text_lengths.dim() == 1, text_lengths.shape
            # Check that batch_size is unified
            assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
            ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
            batch_size = speech.shape[0]




            # for data-parallel
            text = text[:, : text_lengths.max()]
            text_phone=text_phone[:, : text_phone_lengths.max()]
            # 1. Encoder
            encoder_out, encoder_out_lens ,encoder_mask= self.encode(speech, speech_lengths, self.after_norm )
            intermediate_outs = None
            if isinstance(encoder_out, tuple):
                intermediate_outs = encoder_out[1]
                encoder_out = encoder_out[0]


            #先有对齐网络的输出，再是生成网络的输出


            phone_embedding1= self.phone_embedding1.weight.unsqueeze(0).repeat(batch_size,1,1) 

            #需要另外一个网络，去挑选。encoder输出，经过detach之后，拼接在文本上。经过一个transform网络，在音频对应位置，进行ctcloss。
            #文本经过起始符号。
            encoder_mid_out=intermediate_outs[0][1][0]
            mid_out=intermediate_outs[0][1][0]

            text_phone_in_pad,text_phone_out_pad = add_sos_eos(text_phone, 0, 0, 0)
            text_phone_in_pad_flag=text_phone_in_pad
            text_phone_mask=text_phone_in_pad.ge(0).unsqueeze(1)
            text_phone_in_pad=text_phone_in_pad+(~text_phone_in_pad.ge(0))
            text_phonein_lens = text_phone_lengths + 1
            #text_phone 去除-1

            #Phone 乘上embedding
            scores2=torch.nn.functional.one_hot(text_phone_in_pad, num_classes=81).to(torch.float32)
            text_phone_embedding=torch.matmul(scores2 ,phone_embedding1 )
            text_phone_embedding,pos_emb=self.embed(text_phone_embedding)

            #text_phone_in_pad_masked 之后与phone_embedding1相互乘
            text_phone_in_pad_masked=text_phone_in_pad.clone().detach()
            # text_phone_mask2=text_phone_mask
            n = text_phone_in_pad_masked.numel()
            m = int(round(n*0.2))
            indices = np.random.choice(n, m, replace=False) # alternative: indices = torch.randperm(n)[:m]
            text_phone_in_pad_masked.flatten()[indices] = 0
            scores2_mask=torch.nn.functional.one_hot(text_phone_in_pad_masked, num_classes=81).to(torch.float32)
            text_phone_embedding_mask=torch.matmul(scores2_mask ,phone_embedding1 )
            text_phone_embedding_mask,pos_emb1=self.embed(text_phone_embedding_mask)

            #encoder，带有mask的text_encoder输出
            text_phone_embedding_mask, text_phone_mask = self.encoders_add((text_phone_embedding_mask,pos_emb1), text_phone_mask)
            text_phone_embedding_mask=text_phone_embedding_mask[0].detach()
            # text2wav=text_phone_embedding_mask

            #encoder，没有mask的text_encoder输出
            text_phone_embedding, text_phone_mask = self.encoders_add((text_phone_embedding,pos_emb), text_phone_mask)
            text_phone_embedding=text_phone_embedding[0].detach()

            if_encoder_mask=True
            if if_encoder_mask:
                text2wav=self.self_attn(self.after_norm(text_phone_embedding_mask), mid_out, mid_out,encoder_mask)
            else:
                text2wav=self.self_attn(self.after_norm(text_phone_embedding), mid_out, mid_out,encoder_mask)

            wav2text=self.self_attn(mid_out,self.after_norm(text_phone_embedding), self.after_norm(text_phone_embedding), text_phone_mask )
            #是否要增加  mse_loss
            loss_vq_vae=  self.l2_loss(wav2text,  mid_out  ,mask=encoder_mask.transpose(1,2)) 


            #传到share encoder
            
            final_encoder_out=(wav2text,intermediate_outs[0][1][1])
            encoder_out_2, encoder_out_lens ,encoder_mask_= self.encode2(final_encoder_out,  encoder_out_lens )
            encoder_out_text=encoder_out_2[0]
            encoder_out_2_mid=encoder_out_2[1]
            # loss_mse=  self.l2_loss2(encoder_out_text,  encoder_out  ,mask=encoder_mask.transpose(1,2)) 

            style_loss=None
            

            src1=intermediate_outs[1][1][0]
            src2=intermediate_outs[2][1][0]
            src3=intermediate_outs[3][1][0]


            tag1=encoder_out_2_mid[0][1][0]
            tag2=encoder_out_2_mid[1][1][0]
            tag3=encoder_out_2_mid[2][1][0]
       
                
            loss_mse=self.style_loss(src1,tag1,encoder_mask)+self.style_loss(src2,tag2,encoder_mask)+self.style_loss(src3,tag3,encoder_mask)+self.style_loss(encoder_out,encoder_out_text,encoder_mask)
            #loss_mse=self.l2_loss2(src2,tag2,mask=encoder_mask.transpose(1,2))+self.l2_loss2(src3,tag3,mask=encoder_mask.transpose(1,2))+self.l2_loss2(encoder_out,encoder_out_text.detach(),mask=encoder_mask.transpose(1,2))
            #self.l2_loss2(src1,tag1,mask=encoder_mask.transpose(1,2))+
            if if_encoder_mask:
                None
            else:

                #text2wav是否需要在srcattention之后，保留原本text_phone_embedding的一半特征
                old_text=self.after_norm(text_phone_embedding)
                old_mask=text_phone_mask.squeeze(1)
                n = text2wav.shape[0]*text2wav.shape[1]
                m = int(round(n*0.5))
                indices = np.random.choice(n, m, replace=False) # alternative: indices = torch.randperm(n)[:m]
                old_mask1 = old_mask.clone().detach()
                old_mask1.flatten()[indices] = 0
                old_mask2=(~old_mask1)*old_mask
                text2wav=old_mask1.unsqueeze(-1).repeat(1,1,256)*text2wav + old_mask2.unsqueeze(-1).repeat(1,1,256)*old_text
################
            
        
            old_wav=mid_out
            old_mask=encoder_mask.squeeze(1)
            n = wav2text.shape[0]*wav2text.shape[1]
            m = int(round(n*0.5))
            indices = np.random.choice(n, m, replace=False) # alternative: indices = torch.randperm(n)[:m]
            old_mask1 = old_mask.clone().detach()
            old_mask1.flatten()[indices] = 0
            old_mask2=(~old_mask1)*old_mask
            wav2text=old_mask1.unsqueeze(-1).repeat(1,1,256)*wav2text + old_mask2.unsqueeze(-1).repeat(1,1,256)*old_wav



        #     loss_ic, cer_ic = self._calc_ctc_loss_phone(
        # wav2text, encoder_out_lens, text_phone, text_phone_lengths
        # )
            loss_ic, cer_ic = self._calc_ctc_loss_phone(
        mid_out, encoder_out_lens, text_phone, text_phone_lengths
        )
            loss_other_ctc1 = self.criterion_phone( self.ctc_phone.ctc_lo(  text2wav ) , text_phone_in_pad_flag)
            loss_other_ctc2 = self.criterion_phone( self.ctc_phone.ctc_lo(  text_phone_embedding_mask ) , text_phone_in_pad_flag)




            loss_att, acc_att, cer_att, wer_att = None, None, None, None
            loss_ctc, cer_ctc = None, None
            loss_transducer, cer_transducer, wer_transducer = None, None, None
            stats = dict()






            ##???
            # 1. CTC branch
            if self.ctc_weight != 0.0:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

                # Collect CTC branch stats
                stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
                stats["cer_ctc"] = cer_ctc


                loss_ctc2, cer_ctc2 = self._calc_ctc_loss(
                    encoder_out_text, encoder_out_lens, text, text_lengths
                )

                # Collect CTC branch stats
                stats["loss_ctc2"] = loss_ctc2.detach() if loss_ctc2 is not None else None
                stats["cer_ctc2"] = cer_ctc2

            ##???








            ##???
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )
                None



            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = 0.3 * loss_ctc + 0.7 * loss_att + 0.3*loss_ic #+ 0.2*loss_mse  #+0.1*loss_ic + 0.1*loss_other_ctc1 + 0.01*loss_other_ctc2  + 0.1* loss_vq_vae #+  0.1*(loss_mse )+ 0.1* loss_ctc2 #+ 0.1*(loss_vq_vae+ loss_ic_right)   #0.3 * (loss_vq_vae  + loss_ic_right +style_loss+loss_other_ctc11 )+ 

            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att
            stats["loss_vq_vae"] = loss_vq_vae.detach()
            stats["loss_mse"] = loss_mse.detach()


            stats["loss_other_ctc1"] = loss_other_ctc1.detach()
            stats["loss_other_ctc2"] = loss_other_ctc2.detach()
            stats["loss_ic"] = loss_ic.detach()
            # stats["loss_ic2"] = loss_ic2.detach()

            # stats["loss_other_ctc22"] = loss_other_ctc22.detach()
            #stats["style_loss"] = style_loss.detach()
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["loss_ctc"] = loss_ctc.detach()  if loss_ctc is not None else None
            # Collect total loss stats
            stats["loss"] = loss.detach()

            # force_gatherable: to-device and to-tensor if scalar for DataParallel
            loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
            return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, other_normal=None
    ) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, mask = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, mask = self.encoder(feats, feats_lengths,is_old=True,other_normal=other_normal)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens ,mask

        return encoder_out, encoder_out_lens,mask



    def encode2(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        # with autocast(False):
        #     # 1. Extract feats
        #     feats, feats_lengths = self._extract_feats(speech, speech_lengths)

        #     # 2. Data augmentation
        #     if self.specaug is not None and self.training:
        #         feats, feats_lengths = self.specaug(feats, feats_lengths)

        #     # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        #     if self.normalize is not None:
        #         feats, feats_lengths = self.normalize(feats, feats_lengths)

        # # Pre-encoder, e.g. used for raw input data
        # if self.preencoder is not None:
        #     feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)

        feats=speech
        feats_lengths=speech_lengths
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, mask = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, mask = self.encoder(feats, feats_lengths,is_old=False)
        intermediate_outs = None
        # if isinstance(encoder_out, tuple):
        #     intermediate_outs = encoder_out[1]
        #     encoder_out = encoder_out[0]

        # # Post-encoder, e.g. NLU
        # if self.postencoder is not None:
        #     encoder_out, encoder_out_lens = self.postencoder(
        #         encoder_out, encoder_out_lens
        #     )

        # assert encoder_out.size(0) == speech.size(0), (
        #     encoder_out.size(),
        #     speech.size(0),
        # )
        # assert encoder_out.size(1) <= encoder_out_lens.max(), (
        #     encoder_out.size(),
        #     encoder_out_lens.max(),
        # )

        # if intermediate_outs is not None:
        #     return (encoder_out, intermediate_outs), encoder_out_lens ,mask

        return encoder_out, encoder_out_lens,mask


    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc
    def _calc_ctc_loss_phone(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc_phone(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc_phone.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc
    def _calc_ctc_loss_phone_other(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc_phone_other(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc_phone_other.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer
