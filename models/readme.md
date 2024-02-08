# CriticモデルとActorモデルの保存と読み込み

## 保存方法
CriticモデルとActorモデルは、それぞれ`CriticModels`と`ActorModels`というフォルダーに保存されます。
フォルダー内のディレクトリ構成は以下のようになります。


Models/{ObservationSize}/{ActionDim}/critic.pth,actor.pth


ここで、`ObservationSize`は環境からの観測値の次元数、`ActionDim`は行動空間の次元数を表します。
`critic.pth`,`actor.pth`はCriticとActorのモデルの重みを保存したPyTorchのファイルです。

例えば、観測値が3次元で、行動空間が2次元の場合、以下のようにモデルを保存できます。

```python
import torch
# CriticモデルとActorモデルの定義
critic = Critic(3, 2)
actor = Actor(3, 2)
# モデルの重みを保存する
torch.save(critic.state_dict(), "Models/3/2/critic.pth")
torch.save(actor.state_dict(), "Models/3/2/actor.pth")
```

## 読み込み方法
保存したモデルを読み込むには、以下のようにします。

```python

import torch
# CriticモデルとActorモデルの定義
critic = Critic(3, 2)
actor = Actor(3, 2)
# モデルの重みを読み込む
critic.load_state_dict(torch.load("Models/3/2/critic.pth"))
actor.load_state_dict(torch.load("Models/3/2/actor.pth"))
```

これで、保存したモデルを再利用できます。