# Sim2Realを見据えたロボットアーム到達制御の強化学習基盤構築

PyBullet + Gymnasium上に自作したKUKAロボットアーム(7軸)の到達(Reach)タスク環境に対し、SAC(Soft Actor-Critic)をフルスクラッチで実装し、学習・検証を行ったプロジェクトです。比較検証として、PPO(Proximal Policy Optimization)も同様に実装し、同一条件下での性能比較を行いました。

---

## 背景・目的

フィジカルAIの業務フローを想定して実行しました。

```
【データ作成】──>【シミュレーション】──>【実機検証】
```

のうち、ファーストステップにあたる「データ作成〜バーチャルシミュレーション」領域を、7日間の個人学習として一気通貫で実践しました。

着手にあたっては、MDP・ベルマン方程式・TD法・Q学習/SARSA・DQN・Policy Gradient/Actor-Criticに至るまでの強化学習理論を独学でまとめ直し(`docs/rl_notes.pdf`)、その理解を土台にSACの理論整理・実装まで行っています。

---

## 技術構成

| 分類 | 技術 |
|---|---|
| 言語 | Python |
| 深層学習 | PyTorch |
| 環境インターフェース | Gymnasium |
| 物理シミュレーション | PyBullet |
| 可視化 | Matplotlib |
| 開発環境 | Jupyter Notebook, VSCode相当エディタ |

---

## 環境設計:KukaReachEnv

PyBulletに同梱されているKUKA IIWA(7軸ロボットアーム)を用い、Gymnasium形式で到達(Reach)タスク環境をゼロから自作しました。

### タスク内容
エンドエフェクタ(アーム先端)を、エピソードごとにランダム配置されるターゲット位置まで到達させる。

### 観測空間(34次元)

| 内容 | 次元数 | 意図 |
|---|---|---|
| 関節角度 | 7 | 基本情報 |
| 関節角速度 | 7 | 基本情報 |
| エンドエフェクタ位置 | 3 | 基本情報 |
| エンドエフェクタ姿勢(クォータニオン) | 4 | 到達時の向きも考慮した、実践的なタスク設計 |
| ターゲット位置 | 3 | 基本情報 |
| 相対位置ベクトル(target − EE位置) | 3 | 学習の安定化・高速化のため独自に追加 |
| 直前の行動 | 7 | 動きの滑らかさを学習させるため独自に追加(実機を意識) |

### 行動空間(7次元)
各関節への角度指令(連続値、-1〜1に正規化)

### 報酬設計
```
reward = -(エンドエフェクタとターゲットの距離)
distance < 0.05 で成功、成功ボーナスを加算
```

### スコープの絞り込み
Pick and Place(把持・搬送)まで含めると接触判定・成功条件の設計が複雑化し、期限内完成のリスクが高まるため、今回は到達(Reach)タスクに意図的に絞りました。設計上、より複雑なタスクへの拡張は可能です。

---

## アルゴリズム選定理由:なぜDQNではなくSACか

ロボットアームの関節制御は連続値制御であり、離散行動を前提とするDQN(`argmax`によって行動を選択)ではそのまま適用できません。連続値制御に対応した深層強化学習アルゴリズムとして、以下の観点からSACを選定しました。

- **off-policy**:リプレイバッファにより過去の経験を再利用でき、シミュレーションコストが高いロボット制御タスクにおいてサンプル効率で有利
- **最大エントロピー強化学習**:収益に加えてエントロピーも最大化することで、探索を保ちながら学習が進む
- **Twin Q-networkによる過大評価の抑制**、**Polyak平均によるTarget Networkの安定した追従**など、学習を安定させる工夫が理論的に組み込まれている

比較検証として、on-policyの代表的手法であるPPOも実装し、実際に性能を比較しました(後述)。

---

## SAC実装のポイント

- **Actor**:状態から行動の確率分布(平均・標準偏差)を出力。Reparameterization trickにより微分可能な形でサンプリングし、tanhで行動を[-1, 1]に押し込め(squashed Gaussian)、tanh変換に伴う確率密度の補正項を実装
- **Twin Q-network**:独立した2つのCriticを用意し、ターゲット値計算時に小さい方を採用することで、Q値の過大評価を抑制
- **Target Network**:DQNのような定期的なハードコピーではなく、Polyak平均による毎ステップの緩やかな追従を採用
- **エントロピー正則化**:Criticのターゲット値計算・Actorの目的関数の両方に、エントロピー項(`-α logπ`)を組み込み

全て理論から独自にフルスクラッチで実装しています(`sac_agent.py`)。

---

## 学習結果

### ランダム方策 vs SAC学習後

| | 1ステップあたりの平均報酬 |
|---|---|
| ランダム方策 | -0.929 |
| SAC学習後(直近20エピソード平均) | -0.558 |

ランダム方策と比較して、約40%の改善が見られました。

![ランダム方策時の報酬分布・推移](reward_visualization.png)

上図は学習前(ランダム方策)における報酬の分布(左)と、ステップごとの推移(右)です。改善の仕組みを持たないため、報酬は広く分散しています。

![SAC学習曲線](sac_training_progress.png)

学習序盤(0〜25エピソード)で急速に改善し、その後はエントロピー正則化による適度な探索を保ちながら安定した水準で推移しています。

### Critic / Actor Lossの推移

![SAC Loss推移](sac_losses.png)

Critic Lossは急速に収束し安定。Actor Lossは、Criticの評価精度向上に伴い継続的に改善しており、学習プロセス自体が理論通りに健全であることを確認しました。

---

## SAC vs PPO 比較検証

同一環境・同一エピソード数(200エピソード)の条件下で、PPOも実装し比較しました。

![SAC vs PPO](sac_vs_ppo.png)

| | 直近20エピソード平均報酬 |
|---|---|
| SAC | -26.67 |
| PPO | -39.79 |

SACがPPOよりサンプル効率・最終的な収束水準の両方で優位という結果が得られました。これは、off-policy(SAC、データを再利用できる)とon-policy(PPO、収集データを都度使い捨てる)という、両アルゴリズムの理論的特性差を裏付けるものであり、本プロジェクトでSACを主軸に選定した判断の妥当性を実証的に示しています。

PPOはActorCritic一体型ネットワーク、GAE(Generalized Advantage Estimation)、Clipped Surrogate Objectiveを含め、こちらもフルスクラッチで実装しています(`ppo_agent.py`)。

---

## Sim2Realへの工夫:ドメインランダム化

実機との差異(Sim-to-Realギャップ)を見据え、`KukaReachEnv`のリセット時に重力の大きさへ意図的なランダム性(-10.1〜-9.5)を導入しました。単一の固定条件に過学習しない、頑健な方策学習を狙った設計です。導入後も学習の収束性は維持されることを確認しています。

---

## データ収集基盤

ランダム方策によるシミュレーション実行から、状態・行動・報酬・次状態(s, a, r, s')をエピソード単位で収集し、npz形式で保存する基盤を構築しました(`collect_data.py`)。収集したデータの可視化(報酬分布・推移)も実装しています。案件業務フローにおける「データ作成」ステップに対応する取り組みです。

---

## 今後の展望

- **NVIDIA Isaac Lab / Linux環境への移行**:今回はMac環境・期限制約からPyBulletを選定したが、案件で使用されるIsaac SimはNVIDIA GPU・Linux環境が前提となるため、今後同環境への移行を見据えている
- **タスクの拡張**:Reachタスクから、Pick and Placeなどより実践的なタスクへの拡張
- **実機検証**:シミュレーションで学習した方策の、実ロボットでの動作検証とSim2Realギャップの実測

---

## ディレクトリ構成

```
.
├── kuka_reach_env.py         # KukaReachEnv(環境自作)
├── sac_agent.py               # SAC実装(Actor, Twin Critic, ReplayBuffer, SACAgent)
├── ppo_agent.py                # PPO実装(ActorCritic, RolloutBuffer, PPOAgent)
├── collect_data.py             # データ収集パイプライン
├── train_kuka.py               # KukaReachEnv + SAC 学習実行
├── train_kuka_ppo.py           # KukaReachEnv + PPO 学習実行
├── visualize_losses.py         # Loss可視化
├── compare_results.py          # ランダム方策 vs SAC 比較
├── compare_sac_ppo.py          # SAC vs PPO 比較
├── docs/
│   └── rl_notes.pdf            # 強化学習理論ノート(MDP〜SAC)
└── README.md
```

---

## セットアップ

```bash
conda create -n rl_env python=3.11
conda activate rl_env
conda install -c conda-forge pybullet
pip install torch torchvision gymnasium numpy matplotlib h5py
```

```bash
# データ収集
python collect_data.py

# SAC学習
python train_kuka.py

# PPO学習(比較用)
python train_kuka_ppo.py
```
