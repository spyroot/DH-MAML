from typing import Dict, Optional, Tuple
import torch
from torch import nn
from meta_critics.modules.mlp import MLP


class LSTMNet(nn.Module):
    def __init__(self, out_features: int,
                 lstm_kwargs: Dict,
                 mlp_kwargs: Dict,
                 device: Optional[torch.device] = None) -> None:
        """

        :param out_features:
        :param lstm_kwargs:
        :param mlp_kwargs:
        :param device:
        """
        super().__init__()
        lstm_kwargs.update({"batch_first": True})
        self.mlp = MLP(device=device, **mlp_kwargs)
        self.lstm = nn.LSTM(device=device, **lstm_kwargs)
        self.linear = nn.LazyLinear(out_features, device=device)

    def _lstm(self,
              inp: torch.Tensor,
              hidden0_in: Optional[torch.Tensor] = None,
              hidden1_in: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param inp:
        :param hidden0_in:
        :param hidden1_in:
        :return:
        """
        squeeze0 = False
        squeeze1 = False
        if inp.ndimension() == 1:
            squeeze0 = True
            inp = inp.unsqueeze(0).contiguous()

        if inp.ndimension() == 2:
            squeeze1 = True
            inp = inp.unsqueeze(1).contiguous()
        batch, steps = inp.shape[:2]

        if hidden1_in is None and hidden0_in is None:
            shape = (batch, steps) if not squeeze1 else (batch,)
            hidden0_in, hidden1_in = [
                torch.zeros(
                        *shape,
                        self.lstm.num_layers,
                        self.lstm.hidden_size,
                        device=inp.device,
                        dtype=inp.dtype,
                )
                for _ in range(2)
            ]
        elif hidden1_in is None or hidden0_in is None:
            raise RuntimeError(f"got type(hidden0)={type(hidden0_in)} and type(hidden1)={type(hidden1_in)}")
        elif squeeze0:
            hidden0_in = hidden0_in.unsqueeze(0)
            hidden1_in = hidden1_in.unsqueeze(0)

        if not squeeze1:
            _hidden0_in = hidden0_in[:, 0]
            _hidden1_in = hidden1_in[:, 0]
        else:
            _hidden0_in = hidden0_in
            _hidden1_in = hidden1_in
        hidden = (
            _hidden0_in.transpose(-3, -2).contiguous(),
            _hidden1_in.transpose(-3, -2).contiguous(),
        )

        y0, hidden = self.lstm(inp, hidden)
        hidden = tuple(_h.transpose(0, 1) for _h in hidden)
        y = self.linear(y0)

        out = [y, hidden0_in, hidden1_in, *hidden]
        if squeeze1:
            out[0] = out[0].squeeze(1)

        if not squeeze1:
            for i in range(3, 5):
                out[i] = torch.stack([torch.zeros_like(out[i]) for _ in range(inp.shape[1] - 1)] + [out[i]], 1, )
        if squeeze0:
            out = [_out.squeeze(0) for _out in out]
        return tuple(out)

    def forward(self,
                input_tensor: torch.Tensor,
                hidden0_in: Optional[torch.Tensor] = None,
                hidden1_in: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        On forward pass , input tensor passed to mlp -> lstm.
        :param input_tensor:
        :param hidden0_in:
        :param hidden1_in:
        :return:
        """
        x = self.mlp(input_tensor)
        return self._lstm(x, hidden0_in, hidden1_in)
