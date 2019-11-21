import torch
import torch.nn as nn
import torch.nn.functional as F  # pytorch에서 제공해주는 function들을 쓰기 위함

use_cuda = torch.cuda.is_available()  # cuda gpu 사용 가능 여부 확인하여 사용 가능한 경우 True 아니면 False 값 리턴
device = torch.device("cuda" if use_cuda else "cpu")  # 만약 cuda gpu 사용 가능하면 cuda device 사용


class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh # the number of head, 헤드의 개수
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)
        # 일반적인 컨볼루션 레이어입니다. output channel 수가 self.out_channels - self.dv인 이유는 attention output의 channel 수가
        # self.dv이기 때문인데요, 마지막 부분에 일반적인 컨볼루션을 통과해서 나온 출력과, attention을 통과해서 나온 출력을 채널 차원에 대해 concat을
        # 해줍니다. 따라서 전체 output 채널 수는 self.out_channels가 되는 거죠.

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding)
        # attnetion을 위한 컨볼루션 레이어입니다. 여기서 self.dk 앞에 곱하기 2를 해준 건 k와 q의 채널 수가 동일하기 때문입니다.
        # 출력은 그럼 dk dq dv 이렇게 나옴

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)
        # attention의 마지막 출력을 위한 convolution입니다.
        # 입출력 채널이 동일하고, kernel size가 1이면, 출력 채널 즉 dv만큼 커널 하나짜리 웨이트가 있음

        if self.relative:  # SASA와 비슷하게 상대위치에 대한 정보를 인코딩합니다.
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)  # 일반적인 convolution입니다. 마지막 부분에 attention의 출력과 concat하기 위함입니다.
        # 출력 채널은 self.out_channels - self.dv으로 나옴
        batch, _, height, width = conv_out.size()
        # 사이즈는 batch, channel, height, width로 나옴

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh # 채널을 헤더로 나눠서 나오는 것입니다. 
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk) 가 아니고
        # (batch_size, Nh, height, width, dv or dk) 임
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)  # 인풋을 q, k, v,로 나누는데요,
        # 이때 H와 W dimension을 하나로 합치기 때문에 flat이란 말이 붙었습니다. 즉 HxW 부분을 H*W 로 하나의 차원에 몰아넣는 것이죠.

        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)  # q와 k를 행렬곱 줍니다.
        if self.relative:  # 상대위치 정보를 모델이 배울 수 있게 위치 정보를 담고 있는 가중치를 더해주는 작업입니다. 여기서 이 가중치들은 여러 헤드들
            #과 공유됩니다.
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))  # 위에서 구한 weights와 value 값을 행렬곱해줍니다.
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))  # 텐서의 형태를 변경하여줍니다.
        # reshape은 SASA에서 view와 같은 역할을 합니다.

        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)  # 각 헤드(그룹)로 나눠서 처리했던 것을 다시 병합하여 줍니다.
        attn_out = self.attn_out(attn_out)  # attention의 마지막 출력 layer입니다.
        return torch.cat((conv_out, attn_out), dim=1)  # 기존의 convolution output과 attention output을 concat하여 출력합니다.

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)  # q, k, v로 분리하는 부분입니다.
        q = self.split_heads_2d(q, Nh)  # split_heads_2d는 채널 차원은 head의 수로 나누고, head를 위한 차원을 따로 만들어줍니다.
        # 즉, 채널들을 그룹화해주는 것이죠.
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5  # 이 부분은 query를 normalize해주는 부분인데요, 왜 굳이 이 값으로 해주는지는 좀 더 알아봐야 하겠습니다. Normalize가 필요한 이유는, dkh가 클수록 q값도 커지기 때문에 이를 방지하기 위한 수단으로 나눠주는 것입니다.
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))  # H와 W를 하나의 차원에 몰아넣습니다.
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W)) 
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W)) # 샘플 수, 헤드 수, 피처, 높이 * 너비 (높이와 너비가 다른 걸, 둘이 곱해서, 플랫이라고 함)
        return flat_q, flat_k, flat_v, q, k, v # 플랫과, 플랫 전 피처를 반환합니다.

    def split_heads_2d(self, x, Nh): # 피쳐맵의 차원을 변경하여 head의 차원을 포함하게 만들어줍니다.
        batch, channels, height, width = x.size()  # 피쳐맵의 원래 차원인 B x C x H x W
        ret_shape = (batch, Nh, channels // Nh, height, width)  # 새로운 차원인 B x Nh(head의 수) x (C/Nh) x H x W
        # 즉, 원래 피쳐맵의 채널 차원을 분할 하여 개별 head로 만들어주는 것이죠.
        split = torch.reshape(x, ret_shape)  # 여기서 실질적으로 차원 변경이 일어납니다.
        return split

    def combine_heads_2d(self, x):  # 변경했던 피쳐맵의 차원을 다시 돌려주는 함수입니다. 즉, split_heads_2d의 반대 역할을 하는 함수이죠.
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):  # 상대위치에 대한 파라미터를 추가하여 주는 함수입니다.
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)  # 단순히 B x Nh x dk x H x W에서  B x Nh x H x W x dk로 차원의 순서를
        # 바꿔주는 것입니다.

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


# Example Code
# tmp = torch.randn((16, 3, 32, 32)).to(device)
# augmented_conv1 = AugmentedConv(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, Nh=4, relative=True, padding=1, stride=2, shape=16).to(device)
# conv_out1 = augmented_conv1(tmp)
# print(conv_out1.shape)
#
# for name, param in augmented_conv1.named_parameters():
#     print('parameter name: ', name)
#
# augmented_conv2 = AugmentedConv(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, Nh=4, relative=True, padding=1, stride=1, shape=32).to(device)
# conv_out2 = augmented_conv2(tmp)
# print(conv_out2.shape)
