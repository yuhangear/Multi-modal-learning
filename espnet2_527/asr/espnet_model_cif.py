from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from wenet.transformer.vae_decode import vae_decode
import torch
from typeguard import check_argument_types
from torch import Tensor
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

from cif_middleware import CifMiddleware

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield
class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

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
            self.criterion_att2 = LabelSmoothingLoss(
                size=29,
                padding_idx=-1,
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
            self.ctc_phone =  CTC(29, 256)
            self.ctc_phone_other=  CTC(29, 256)
        self.phone_embedding1 = torch.nn.Embedding(
            num_embeddings=29,  # the last for PAD valu衰减直。这里的大小设置的是整个词典的大小。
            embedding_dim=256,
          )


        self.after_norm = torch.nn.LayerNorm(256)
        self.norm1=torch.nn.LayerNorm(256)
        # self.vector_quantizer = GumbelVectorQuantizer(
        #         dim=256,
        #         num_vars=args.vq_vars,
        #         temp=eval(args.vq_temp),
        #         groups=args.vq_groups,
        #         combine_groups=args.combine_groups,
        #         vq_dim=args.vq_dim if args.vq_dim > 0 else embed,
        #         time_first=False,
        #         activation=activation,
        #         weight_proj_depth=args.vq_depth,
        #         weight_proj_factor=2,
        #     )

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.line_emb1=torch.nn.Linear(256, 29)



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
        self.change_embedding1=EncoderLayer_add(
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
        self.change_embedding2=EncoderLayer_add(
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
        self.change_embedding3=EncoderLayer_add(
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
        self.norm1=torch.nn.LayerNorm(256)
        self.norm2=torch.nn.LayerNorm(256)
        self.norm3=torch.nn.LayerNorm(256)

        self.old_feat=None
        self.old_feat_length=None
        cgf={
            "cif_embedding_dim" : 256 ,
            "encoder_embed_dim" : 256 ,
            "produce_weight_type" : "conv",
            "cif_threshold" : 0.99,
            "conv_cif_layer_num" : 1,
            "conv_cif_width" :  5,
            "conv_cif_output_channels_num" : 256,
            "conv_cif_dropout" : 0.1,
            "dense_cif_units_num" : 256,
            "apply_scaling" : True,
            "apply_tail_handling" : True,
            "tail_handling_firing_threshold" : 0.5,
            "add_cif_ctxt_layers" : False,
        }
        cgf= obj(cgf)
        self.cif=CifMiddleware(          cgf     )
        self.l1_loss = nn.L1Loss()
        self.vae_decode=vae_decode()
        self.batch_norm = torch.nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_phone: torch.Tensor,
        text_phone_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        if len(speech[0])==5 and speech[0][0]==1.0:



            batch_size=speech.shape[0]
            #phone_embedding1=(self.phone_embedding1.weight+self.phone_embedding2.weight+self.phone_embedding3.weight+self.phone_embedding4.weight+self.phone_embedding5.weight+self.phone_embedding6.weight+self.phone_embedding7.weight+self.phone_embedding8.weight)/8
            phone_embedding1= self.phone_embedding1.weight.unsqueeze(0).repeat(batch_size,1,1) 
            #phone_embedding=torch.cat((self.phone_embedding1.weight.unsqueeze(0)  ,self.phone_embedding2.weight.unsqueeze(0)  ,self.phone_embedding3.weight.unsqueeze(0)  ,self.phone_embedding4.weight.unsqueeze(0)  ,self.phone_embedding5.weight.unsqueeze(0)  ,self.phone_embedding6.weight.unsqueeze(0)  ,self.phone_embedding7.weight.unsqueeze(0)  ,self.phone_embedding8.weight.unsqueeze(0)  ),0)
            #phone_embedding= phone_embedding1.unsqueeze(0).repeat(batch_size,1,1).transpose(1,2)
            phone_embedding=phone_embedding1

            device=text.device
            phone_len=text_phone.shape[1]
            
            #B:T:29
            # scores = 0.0000000001*torch.ones(batch_size,phone_len,29).to(device)
            # for B in range(text_phone.shape[0]):
            #     for index in range(text_phone[B].shape[0]):
            #         phone_name=text_phone[B][index]
            #         if phone_name!=-1:
            #             scores[B][index][phone_name]=0.9999

            encoder_out_lens=text_phone_lengths
            encoder_mask=text_phone.ge(0).unsqueeze(1)

            text_phone2=text_phone+(~encoder_mask.squeeze(1)).to(torch.long).detach()
            scores=torch.nn.functional.one_hot(text_phone2, num_classes=29).to(torch.float32)

            #p_attn = self.dropout(scores)

            cross_attention_out = torch.matmul(scores, phone_embedding)



            # for data-parallel
            text = text[:, : text_lengths.max()]
            text_phone=text_phone[:, : text_phone_lengths.max()]

        
            cross_attention_out,pos_emb=self.encoder.embed.out[1](cross_attention_out)
            # cross_attention_out=(cross_attention_out,pos_emb)
            # 1. Encoder
            final_encoder_out=      F.relu( self.line_phone(cross_attention_out))

            final_encoder_out, chunk_masks  = self.change_embedding1((final_encoder_out,pos_emb),  encoder_mask)
        
    
    ###
            final_encoder_out=self.norm1(final_encoder_out[0])
            final_encoder_out, chunk_masks  = self.change_embedding2((final_encoder_out, pos_emb),  encoder_mask)
            final_encoder_out=self.norm1(final_encoder_out[0])
            final_encoder_out, chunk_masks  = self.change_embedding3((final_encoder_out, pos_emb),  encoder_mask)
    ###




            encoder_out, encoder_out_lens ,encoder_mask= self.encode2((final_encoder_out[0],final_encoder_out[1]), encoder_out_lens)




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

            # Intermediate CTC (optional)
            loss_interctc = 0.0
            if self.interctc_weight != 0.0 and intermediate_outs is not None:
                for layer_idx, intermediate_out in intermediate_outs:
                    # we assume intermediate_out has the same length & padding
                    # as those of encoder_out
                    loss_ic, cer_ic = self._calc_ctc_loss_phone(
                        self.after_norm(intermediate_out[0]), encoder_out_lens, text_phone, text_phone_lengths
                    )

                    loss_interctc = loss_interctc + loss_ic

                    # Collect Intermedaite CTC stats
                    stats["loss_interctc_layer{}".format(layer_idx)] = (
                        loss_ic.detach() if loss_ic is not None else None
                    )
                    stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

                    #other
                    #################
                    # we assume intermediate_out has the same length & padding
                    # as those of encoder_out

        

                    #中间层CTC，有 原本的encoder 输出，还有final_en_
                    loss_interctc = loss_interctc





                loss_interctc = loss_interctc / len(intermediate_outs)


                loss_ctc = 0.5 * loss_ctc + 0.5 * loss_interctc  


            if self.use_transducer_decoder:
                # 2a. Transducer decoder branch
                (
                    loss_transducer,
                    cer_transducer,
                    wer_transducer,
                ) = self._calc_transducer_loss(
                    encoder_out,
                    encoder_out_lens,
                    text,
                )

                if loss_ctc is not None:
                    loss = loss_transducer + (self.ctc_weight * loss_ctc)
                else:
                    loss = loss_transducer

                # Collect Transducer branch stats
                stats["loss_transducer"] = (
                    loss_transducer.detach() if loss_transducer is not None else None
                )
                stats["cer_transducer"] = cer_transducer
                stats["wer_transducer"] = wer_transducer

            else:
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

                    loss =0.3 * loss_ctc + 0.7 * loss_att 

                # Collect Attn branch stats
                stats["loss_att"] = loss_att.detach() if loss_att is not None else None
                stats["acc"] = acc_att
                stats["cer"] = cer_att
                stats["wer"] = wer_att

            # Collect total loss stats
            stats["loss"] = loss.detach()

            # force_gatherable: to-device and to-tensor if scalar for DataParallel
            loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
            return loss, stats, weight






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
            encoder_out, encoder_out_lens ,encoder_mask= self.encode(speech, speech_lengths)
            intermediate_outs = None
            if isinstance(encoder_out, tuple):
                intermediate_outs = encoder_out[1]
                encoder_out = encoder_out[0]




            phone_embedding1= self.phone_embedding1.weight.unsqueeze(0).repeat(batch_size,1,1) 


            phone_mask=text_phone.ge(0).unsqueeze(1)

            #直接由文本得到对齐。
            #text_phone_lengths有了
            text_phone2=text_phone+(~phone_mask.squeeze(1)).to(torch.long).detach()
            scores=torch.nn.functional.one_hot(text_phone2, num_classes=29).to(torch.float32)
            cross_attention_out=torch.matmul(scores ,phone_embedding1 )

            cross_attention_out,pos_emb=self.encoder.embed.out[1](cross_attention_out)


            final_encoder_out=      F.relu( self.line_phone(cross_attention_out))

            final_encoder_out, chunk_masks  = self.change_embedding1((final_encoder_out,pos_emb),  phone_mask)

            final_encoder_out=self.norm1(final_encoder_out[0])

            final_encoder_out, chunk_masks  = self.change_embedding2((final_encoder_out,  pos_emb),  phone_mask)

            final_encoder_out=self.norm2(final_encoder_out[0])
            final_encoder_out, chunk_masks  = self.change_embedding3((final_encoder_out,  pos_emb),  phone_mask)
            final_encoder_out=final_encoder_out[0]
            l2_loss = seqloss.MaskedMSELoss()


            conv_input = intermediate_outs[0][1][0].permute(0, 2, 1)
            conv_out = self.cif.conv(conv_input)
            proj_input = conv_out.permute(0, 2, 1)
            proj_input = self.cif.conv_dropout(proj_input)
            sig_input = self.cif.weight_proj(self.batch_norm(proj_input.transpose(1,2)).transpose(1,2))
            weight = torch.sigmoid(sig_input).squeeze(2)
            pk=(encoder_mask).squeeze(1)
            weight=weight*pk

            # Apply scaling strategies
        
            # Conduct scaling when training
            # weight_sum = weight.sum(-1)             # weight_sum has shape B
            # normalize_scalar = torch.unsqueeze(
            #     text_phone_lengths / weight_sum, -1)    # normalize_scalar has shape B x 1
            # weight = weight * normalize_scalar
            
            cif_out=cif_function(intermediate_outs[0][1][0],weight,beta=0.99,tail_thres=0.5,padding_mask=(~encoder_mask).squeeze(1),target_lengths=text_phone_lengths)
            cif_encode_mid=cif_out["cif_out"][0]
            alpha_sum=cif_out["alpha_sum"]
            loss_a_len=  self.l1_loss(alpha_sum[0],text_phone_lengths.to(torch.float16))

            criterion = torch.nn.CrossEntropyLoss()

            # c1=criterion(self.ctc_phone.ctc_lo(self.after_norm(final_encoder_out)).softmax(-1), scores.detach())
            # c2=criterion(self.ctc_phone.ctc_lo(self.after_norm(cif_encode_mid)).softmax(-1), scores.detach())

            c1=self.criterion_att2(self.ctc_phone.ctc_lo(self.after_norm(final_encoder_out)),text_phone.detach())

            # c2=criterion(self.ctc_phone.ctc_lo(self.after_norm(cif_encode_mid)).softmax(-1), scores.detach())
            c2=self.criterion_att2(self.ctc_phone.ctc_lo(self.after_norm(cif_encode_mid)), text_phone.detach())

            c3=0.2* c1+0.8* c2
            loss_vq_vae=  l2_loss(final_encoder_out,  cif_encode_mid  ,mask=phone_mask.transpose(1,2)) 
            all_other_loss= loss_vq_vae + c3 +  loss_a_len   #+  0.1*l2_loss(final_encoder_out.clone().detach(), intermediate_outs[0][1][0],mask=attn2_temp_mask.transpose(1,2))



            
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
            loss_ctc, cer_ctc = None, None
            loss_transducer, cer_transducer, wer_transducer = None, None, None
            stats = dict()




            #人为构造中间层输出，传导后续的模型当中
            #加一个if判断，
            #xs_pad, masks=self.encoder.encoders[9](encoder_out[1][0][1],encoder_mask)




            # 1. CTC branch
            if self.ctc_weight != 0.0:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

                # Collect CTC branch stats
                stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
                stats["cer_ctc"] = cer_ctc

            # Intermediate CTC (optional)
            loss_interctc = 0.0
            if self.interctc_weight != 0.0 and intermediate_outs is not None:
                for layer_idx, intermediate_out in intermediate_outs:
                    # we assume intermediate_out has the same length & padding
                    # as those of encoder_out
                    loss_ic, cer_ic = self._calc_ctc_loss_phone(
                        self.after_norm(intermediate_out[0]), encoder_out_lens, text_phone, text_phone_lengths
                    )

                    loss_interctc = loss_interctc + loss_ic

                    # Collect Intermedaite CTC stats
                    stats["loss_interctc_layer{}".format(layer_idx)] = (
                        loss_ic.detach() if loss_ic is not None else None
                    )
                    stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

                    #other
                    #################
                    # we assume intermediate_out has the same length & padding
                    # as those of encoder_out

        

                    #中间层CTC，有 原本的encoder 输出，还有final_en_
                    loss_interctc = loss_interctc





                loss_interctc = loss_interctc / len(intermediate_outs)


                loss_ctc = 0.5 * loss_ctc + 0.5 * loss_interctc  


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

                loss =0.3 * loss_ctc + 0.5 * loss_att +0.3* all_other_loss

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att
            stats["loss_vq_vae"] = loss_vq_vae.detach()
            stats["loss_a_len"]=loss_a_len.detach()
            stats["c2"]=c2.detach()
            stats["c1"]=c1.detach()
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
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
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
            encoder_out, encoder_out_lens, mask = self.encoder(feats, feats_lengths,is_old=True)
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


def prob_check(tensor, eps=1e-10, neg_inf=-1e8, logp=False):
    assert not torch.isnan(tensor).any(), (
        "Nan in a probability tensor."
    )
    # Add the eps here to prevent errors introduced by precision
    if logp:
        assert tensor.le(0).all() and tensor.ge(neg_inf).all(), (
            "Incorrect values in a log-probability tensor"
            ", -inf <= tensor <= 0"
        )
    else:
        assert tensor.le(1.0 + eps).all() and tensor.ge(0.0 - eps).all(), (
            "Incorrect values in a probability tensor"
            ", 0.0 <= tensor <= 1.0"
        )


def cif_function(
    input: Tensor,
    alpha: Tensor,
    beta: float = 1.0,
    tail_thres: float = 0.5,
    padding_mask: Optional[Tensor] = None,
    target_lengths: Optional[Tensor] = None,
    eps: float = 1e-4,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r""" A fast parallel implementation of continuous integrate-and-fire (CIF)
    https://arxiv.org/abs/1905.11235
    Args:
        input (Tensor): (N, S, C) Input features to be integrated.
        alpha (Tensor): (N, S) Weights corresponding to each elements in the
            input. It is expected to be after sigmoid function.
        beta (float): the threshold used for determine firing.
        tail_thres (float): the threshold for determine firing for tail handling.
        padding_mask (Tensor, optional): (N, S) A binary mask representing
            padded elements in the input.
        target_lengths (Tensor, optional): (N,) Desired length of the targets
            for each sample in the minibatch.
        eps (float, optional): Epsilon to prevent underflow for divisions.
            Default: 1e-4
    Returns -> Dict[str, List[Optional[Tensor]]]: Key/values described below.
        cif_out (Tensor): (N, T, C) The output integrated from the source.
        cif_lengths (Tensor): (N,) The output length for each element in batch.
        alpha_sum (Tensor): (N,) The sum of alpha for each element in batch.
            Can be used to compute the quantity loss.
        delays (Tensor): (N, T) The expected delay (in terms of source tokens) for
            each target tokens in the batch.
        tail_weights (Tensor, optional): (N,) During inference, return the tail.
    """
    B, S, C = input.size()
    assert tuple(alpha.size()) == (B, S), f"{alpha.size()} != {(B, S)}"
    prob_check(alpha)

    dtype = alpha.dtype
    alpha = alpha.float()
    if padding_mask is not None:
        padding_mask = padding_mask.bool()
        alpha = alpha.masked_fill(padding_mask, 0)

    if target_lengths is not None:
        feat_lengths = target_lengths.long()
        desired_sum = beta * target_lengths.type_as(input) + eps
        alpha_sum = alpha.sum(1)
        alpha = alpha * (desired_sum / alpha_sum).unsqueeze(1)
        T = feat_lengths.max()
    else:
        alpha_sum = alpha.sum(1)
        feat_lengths = (alpha_sum / beta).floor().long()
        T = feat_lengths.max()

    # aggregate and integrate
    csum = alpha.cumsum(-1)
    with torch.no_grad():
        # indices used for scattering
        right_idx = (csum / beta).floor().long().clip(max=T)
        left_idx = right_idx.roll(1, dims=1)
        left_idx[:, 0] = 0

        # count # of fires from each source
        fire_num = right_idx - left_idx
        extra_weights = (fire_num - 1).clip(min=0)

    # The extra entry in last dim is for
    output = input.new_zeros((B, T + 1, C))
    delay = input.new_zeros((B, T + 1))
    source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(input)
    zero = alpha.new_zeros((1,))

    # right scatter
    fire_mask = fire_num > 0
    right_weight = torch.where(
        fire_mask,
        csum - right_idx.type_as(alpha) * beta,
        zero
    ).type_as(input)
    # assert right_weight.ge(0).all(), f"{right_weight} should be non-negative."
    output.scatter_add_(
        1,
        right_idx.unsqueeze(-1).expand(-1, -1, C),
        right_weight.unsqueeze(-1) * input
    )
    delay.scatter_add_(
        1,
        right_idx,
        right_weight * source_range / beta
    )

    # left scatter
    left_weight = (
        alpha - right_weight - extra_weights.type_as(alpha) * beta
    ).type_as(input)
    output.scatter_add_(
        1,
        left_idx.unsqueeze(-1).expand(-1, -1, C),
        left_weight.unsqueeze(-1) * input
    )
    delay.scatter_add_(
        1,
        left_idx,
        left_weight * source_range / beta
    )

    # extra scatters
    if extra_weights.ge(0).any():
        extra_steps = extra_weights.max().item()
        tgt_idx = left_idx
        src_feats = input * beta
        for _ in range(extra_steps):
            tgt_idx = (tgt_idx + 1).clip(max=T)
            # (B, S, 1)
            src_mask = (extra_weights > 0)
            output.scatter_add_(
                1,
                tgt_idx.unsqueeze(-1).expand(-1, -1, C),
                src_feats * src_mask.unsqueeze(2)
            )
            delay.scatter_add_(
                1,
                tgt_idx,
                source_range * src_mask
            )
            extra_weights -= 1

    # tail handling
    if target_lengths is not None:
        # training time -> ignore tail
        output = output[:, :T, :]
        delay = delay[:, :T]
    else:
        # find out contribution to output tail
        # note: w/o scaling, extra weight is all 0
        zero = right_weight.new_zeros((1,))
        r_mask = right_idx == feat_lengths.unsqueeze(1)
        tail_weights = torch.where(r_mask, right_weight, zero).sum(-1)
        l_mask = left_idx == feat_lengths.unsqueeze(1)
        tail_weights += torch.where(l_mask, left_weight, zero).sum(-1)

        # a size (B,) mask that extends position that passed threshold.
        extend_mask = tail_weights >= tail_thres

        # extend 1 fire and upscale the weights
        if extend_mask.any():
            # (B, T, C), may have infs so need the mask
            upscale = (
                torch.ones_like(output)
                .scatter(
                    1,
                    feat_lengths.view(B, 1, 1).expand(-1, -1, C),
                    beta / tail_weights.view(B, 1, 1).expand(-1, -1, C),
                )
            )
            output[extend_mask] *= upscale[extend_mask]
            feat_lengths += extend_mask.long()
            T = feat_lengths.max()
        output = output[:, :T, :]
        delay = delay[:, :T]

        # a size (B, T) mask to erase weights
        tail_mask = torch.arange(T, device=output.device).unsqueeze(0) >= feat_lengths.unsqueeze(1)
        output[tail_mask] = 0

    return {
        "cif_out": [output],
        "cif_lengths": [feat_lengths],
        "alpha_sum": [alpha_sum.to(dtype)],
        "delays": [delay],
        "tail_weights": [tail_weights] if target_lengths is None else []
    }