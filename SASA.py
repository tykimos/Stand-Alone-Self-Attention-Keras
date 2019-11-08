import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups  # 여기서 groups의 수는 들어오는 인풋을 몇 개의 group으로 나눠서 컨볼루션 진행할 거냐를 결정합니다. 여기서 각 나뉘어진
        # 그룹을 헤드(head)라고 부릅니다.

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
        # (당연히) 나가는 채널의 수가 처리하는 헤드 수로 나눌 수 있어야 합니다. 만약 그렇지 않으면 에러를 냅니다.

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)
        # 여기서 위 두 변수는 중심이 되는 픽셀에 대하여 상대 위치(height와 width)를 정하는 파라미터인데요, 역시 훈련 가능한 가중치입니다.
        # 아시다시피 우리가 컨볼루션을 할 때 kernel을 왼쪽 위에서부터 오른쪽 아래로 움직이면서 계산을 하는데요, 중심이 되는 픽셀이란 그 kernel의 중심이
        # 되는 부분에 해당하는 픽셀을 의미합니다. 때문에 그 중심 픽셀에 대해 상대 위치를 표현하기 위해서는 kernel size 만큼의 parameter가
        # 필요한 것이죠. 예를 들면, kernel size가 3x3인 경우, 다음과 같이 상대 위치를 표현할 수 있습니다. 중심이 되는 픽셀의 위치가 (0, 0) 입니다.
        # (-1, -1) (0, -1) (1, -1)
        # (-1, 0)  (0, 0)  (1, 0)
        # (-1, 1)  (0, 1)  (1, 1)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        # 인풋을 key, query, 그리고 value 각각에 대해 1x1 convolution으로 연산합니다.
        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])  # 왼쪽 오른쪽 위 아래에 대해 패딩해줍니다.
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # unfold는 인풋을 명시된 차원에 대해 잘라주는 역할을 합니다. 예를 들면, 다음과 같습니다.
        # x = torch.arange(1., 8)
        # tensor([1., 2., 3., 4., 5., 6., 7.])
        # x.unfold(0, 2, 1)
        # tensor([[1., 2.],
        #         [2., 3.],
        #         [3., 4.],
        #         [4., 5.],
        #         [5., 6.],
        #         [6., 7.]])
        # x.unfold(0, 2, 2)
        # tensor([[1., 2.],
        #         [3., 4.],
        #         [5., 6.]])
        # 파이토치의 경우 텐서의 형태가 배치x채널x세로x가로이기 때문에 2차원과 3차원에 대해서 unfold한다는 건 이미지의 가로 세로를 잘라서 저장하겠다는
        # 겁니다. 이 부분이 약간 헷갈리실 수 있는데요, 가로 세로가 5x5인 인풋을 3x3으로 자른다고 한다면 보폭이 1일 때 왼쪽 위부터 오른쪽 아래까지 총
        # 9개의 3x3 이미지가 나오겠죠? 그걸 분리해서 저장하겠다는 의미입니다.

        v_out_h, v_out_w = v_out.split(self.out_channels // 2, dim=1)  # value out 채널을 둘로 나누는 작업입니다.
        v_out = torch.cat((v_out_h + self.rel_h, v_out_w + self.rel_w), dim=1)  # 그리고 나눠진 두 채널에 하나는 가로에 대한
        # 상대위치 가중치를 더하고, 나머지에는 세로에 대한 상대위치 가중치를 더하여 다시 합칩니다. 이 부분을 나눠서 하는 이유는 좀 더 고민해 봐야 할 것
        # 같습니다. ^^;

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        # key와 value의 텐서 형태를 바꾸는 작업(view)입니다. 여기서 주의할 건 텐서를 이루고 있는 요소 수는 변하지 않습니다. 예를 들어 2x3의 경우
        # 1x6 또는 3x2와 같이 변경할 수 있지만 2x4처럼 요소 수가 변할 수는 없다는 것이죠.

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1) # q와 k를 곱한 값에 softmax를 해주는 이유는 중심이 되는 픽셀인 q와 그 이웃한 픽셀들이 얼마나 관련이
        # 있는지를 확률적으로 나타내기 위해서입니다. 위에서 q는 1x1 convolution만 해주고, k의 경우에는 kernel size만큼 이미지를 잘라서 저장한
        # 사실을 기억해주시기 바랍니다. 즉, q는 인풋의 각 픽셀에 대한 정보를, k는 각 픽셀을 포함한 주변 정보를 담고 있는 셈이죠. :)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        # einsum은 Einstein summation의 약자인데요, 텐서의 차원이 여러 개일 때 어떤 차원에 대해서 행렬곱을 해줄지 지정해주는 것에 불과합니다.
        # 여기서는 'bnchwk'라는 6개의 차원 중에 k라는 차원에 대해 행렬곱을 해달라고 하는 것이죠. view는 위에 나왔던 것과 마찬가지로 텐서의 형태를
        # 변경하겠다 하는 것입니다.

        return out

    def reset_parameters(self):
        # 아래는 parameter의 초기값을 정해주는 역할을 합니다.
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

# AttentionStem은 전체 network의 가장 처음 연산하는 곳에 적용되는 부분입니다. 이 부분을 AttentionConv와 별도로 만들어준 이유는 classification
# model의 처음 부분이 하는 역할과 나중 부분이 하는 역할이 다르기 때문입니다. Classification 모델의 처음 부분은 edge detector의 역할을 하는 반면
# 나중 부분은 abstract한 feature를 추출하죠.

class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m  # m(multiple)개 만큼 value matrix를 만듭니다.

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)
        # emb_a, emb_b, emb_mix는 AttentionConv와 비슷하게 상대위치를 결정해주는 역할을 합니다.

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])
        # value matrix를 m개 만큼 만드는 convolution을 정의합니다.

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)
        # 이 작업은 AttentionConv에서 했던 것과 동일하죠. 가로 세로를 kernel 사이즈 만큼 개별로 잘라서 보관하는 작업입니다.

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]  # k_out보다 차원이 하나 더 많은 것은 value matrix를 m개 만큼 생성했기
        # 때문입니다. 여기서 0번째 차원이 m값을 가집니다.

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        # unsqueeze는 인자로 받은 차원을 만들어주는 역할을 합니다. 즉 emb_logit_a의 경우에는 m x a x 1,
        # emb_logit_b의 경우에는 m x 1 x b의 형태를 갖게 됩니다. 여기서 a, b는 k(kernel size)와 동일합니다.

        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)
        # emb의 형태를 m x 1 x 1 x 1 x 1 x k x k 의 형태로 바꾼 후, softmax를 취해줍니다. 형태를 저렇게 바꿔주는 이유는 v_out과 행렬곱을
        # 하게 하기 위함입니다.

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        # v_out의 0번째 차원은 처음에 만든 m개의 value matrix를 포함하고 있는데요, 0번째 차원을 더함으로써 그 차원을 없앱니다.

        # 아래 부분은 AttentionConv에서 하는 연산과 동일합니다.
        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        # 가중치를 초기화 하는 부분입니다.
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)


# temp = torch.randn((2, 3, 32, 32))
# conv = AttentionConv(3, 16, kernel_size=3, padding=1)
# print(conv(temp).size())