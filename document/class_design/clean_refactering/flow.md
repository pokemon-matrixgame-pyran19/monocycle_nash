
## 概要
全体的に依存の流れがおかしい。
run_xxxx系の関数はアプリケーション層にあたる部分なのでインフラ層のconfig_loaderに依存するべきではない。
他の部分もなんか怪しいところ多いので作り直す。


- 行列の変形とか均衡解のsolverはドメイン層にあたるので、featureの処理の流れ(アプリケーション層)部分の処理に依存しないように作る。(ここに関しては既にきちんと出来てる)
- 各featureのrunが呼び出されるとそのfeature用のコンフィグを読み込んでその内容に従って処理を行う。(アプリケーション層)
- コンフィグからの読み込みはアプリケーション層で抽象を定義してそれを呼び出す。インフラ層で読み込みの詳細を実装する。複数のコンフィグファイルを合成するみたいな部分はインフラ側の事情なのでアプリ側には隠ぺいする。
- 行列をファイルから読み込む処理などは現在アプリケーション層でファイルを読み込み→インプットファイルで行列の形式に書いていることを信頼してドメイン層の初期化メソッドに渡す、みたいになっている。これを①インフラ層に行列読み込みのリポジトリを作る②アプリケーション層では読み込みリポジトリを呼び出す③アプリケーション層ではインフラ層が読む行列がどれかを指定(featureによってはreferrence matrixかsource matrixかなど複数種類の行列があったりするのでそれを指定) みたいにする。


例:
``` 現在の実装.py
def run(config_loader: MainConfigLoader) -> int:
    loaded = config_loader.load_inputs_for_feature(FEATURE_NAME)
    approximation_data = loaded.approximation_data
    source_matrix_data, reference_matrix_data = _load_source_and_reference_matrices(config_loader, loaded.matrix_data, approximation_data)
```


``` やりたいイメージ.py
class MatrixReader(ABC):
    def read(path):
       pass # インフラ層で具体的に実装. pathのファイルから読み込み

class ThisFeatureConfig():
    # このfeature用のコンフィグを色々記述


class ThisFeatureConfigReader(ABC):
    def read():
    # このfeature特有の設定の読み込み処理をインフラ層で具体的に実装
    pass

def run(config_reader: ThisFeatureConfigReader, matrix_reader: MatrixReader) -> int:
    config=config_reader.read()
    source_matrix = matrix_reader.read(config.source_path)
    referrence_matrix = matrix_reader.read(config.refference_path)
```
