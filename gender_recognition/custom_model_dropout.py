# to install speech brain, have to run brew install cmake first on M1 chip

import torch  # noqa: F401
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.dropout import Dropout2d
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet import activations, containers


class Xvector(torch.nn.Module):
    """This model extracts X-vectors for speaker recognition

    Arguments
    ---------
    activation : torch class
        A class for constructing the activation layers.
    tdnn_blocks : int
        Number of time-delay neural (TDNN) layers.
    tdnn_channels : list of ints
        Output channels for TDNN layer.
    tdnn_kernel_sizes : list of ints
        List of kernel sizes for each TDNN layer.
    tdnn_dilations : list of ints
        List of dilations for kernels in each TDNN layer.
    lin_neurons : int
        Number of neurons in linear layers.

    Example
    -------
    >>> compute_xvect = Xvector()
    >>> input_feats = torch.rand([5, 10, 40])
    >>> outputs = compute_xvect(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 512])
    """

    def __init__(
        self,
        device="cpu",
        activation=torch.nn.LeakyReLU,
        tdnn_blocks=5,
        tdnn_channels=[512, 512, 512, 512, 1500],
        tdnn_kernel_sizes=[5, 3, 3, 1, 1],
        tdnn_dilations=[1, 2, 3, 1, 1],
        lin_neurons=512,
        in_channels=40,
        final_dropout=0.1
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        # TDNN has convolutional layers with the given dilation factors
        # and kernel sizes. We here loop over all the convolutional layers
        # that we wanna add. Note that batch normalization is used after
        # the activations function in this case. This improves the
        # singer identification performance a bit.
        for block_index in range(tdnn_blocks):
            out_channels = tdnn_channels[block_index]
            self.blocks.extend(
                [
                    Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=tdnn_kernel_sizes[block_index],
                        dilation=tdnn_dilations[block_index],
                    ).requires_grad_(block_index>1),
                    activation(),
                    BatchNorm1d(input_size=out_channels).requires_grad_(block_index>1),
                ]
            )
            in_channels = tdnn_channels[block_index]

        # Statistical pooling. It converts a tensor of variable length
        # into a fixed-length tensor. The statistical pooling returns the
        # mean and the standard deviation.
        self.blocks.append(StatisticsPooling())

        # Final linear transformation.
        self.blocks.append(
            Linear(
                input_size=out_channels * 2,  # mean + std,
                n_neurons=lin_neurons,  # number of neurons of linear layer
                bias=True,
                combine_dims=False,
            )
        )

        self.blocks.append(
            Dropout2d(drop_rate=final_dropout)
        )


    def forward(self, x, lens=None):
        """Returns the x-vectors.

        Arguments
        ---------
        x : torch.Tensor
        """

        for layer in self.blocks:
            try:
                x = layer(x, lengths=lens)
            except TypeError:
                x = layer(x)
        return x


class Classifier(sb.nnet.containers.Sequential):
    """This class implements the last MLP on the top of xvector features.
    Arguments
    ---------
    input_shape : tuple
        Expected shape of an example input.
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of output neurons.

    Example
    -------
    >>> input_feats = torch.rand([5, 10, 40])
    >>> compute_xvect = Xvector()
    >>> xvects = compute_xvect(input_feats)
    >>> classify = Classifier(input_shape=xvects.shape)
    >>> output = classify(xvects)
    >>> output.shape
    torch.Size([5, 1, 1211])
    It returns a tensor of one hot encoding for 1211 different speakers(will change in my case)
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=2,
        dropout=0.3
    ):
        super().__init__(input_shape=input_shape)

        self.append(activation(), layer_name="act")
        self.append(sb.nnet.normalization.BatchNorm1d, layer_name="norm")

        if lin_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")

        # Adding fully-connected layers
        for block_index in range(lin_blocks):

            # for each fc layer, I have first a linear layer, then an activation layer and then a batchnorm layer
            block_name = f"block_{block_index}"
            self.DNN.append(
                sb.nnet.containers.Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                sb.nnet.linear.Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(activation(), layer_name="act")
            self.DNN[block_name].append(
                sb.nnet.normalization.BatchNorm1d, layer_name="norm"
            )
            self.DNN[block_name].append(
                Dropout2d(drop_rate=dropout)
            )

        # Final Softmax classifier
        # Add these after all the DNN layers
        self.append(
            sb.nnet.linear.Linear, n_neurons=out_neurons, layer_name="out"
        )
        self.append(
            sb.nnet.activations.Softmax(apply_log=True), layer_name="softmax"
        )
