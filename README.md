# CheckpointAutoRemove
### Checkpointファイルの自動削除コードサンプル

学習時の中間モデル保存時、  
「{epoch:03d}-{val_loss:.5f}.hdf5」のようなデータがいっぱいでき上がってしまう。  
ちょうど「Best３だけ残す」みたいな事ができればいいのに！  
という事を実現する方法を考えました。  

https://maxigundan.com/deeplearning/?p=67

↑ 詳しい説明はこちら


### 実行方法

`python train_and_autoremove.py`

→ CIFAR10分類を利用した、カンタンな学習コードをサンプルとしています。  
Checkpointフォルダに、学習モデルが保存され、  
最新ファイル３つのみ保持、古いファイルが自動で削除されます。 

### コンフィグ
CheckpointToolsにわたす'num_saves'の値を変更することで  
残す数を指定できます。
デフォルトだと、'3'
