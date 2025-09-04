# 2024 部品製造第二課 内田
"""パッチコア関連モジュール."""
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
import cv2


def preprocess_images(images: np.ndarray, shape=(640, 640)) -> np.ndarray:
    """画像データを前処理し、(N, H, W, C)のfloat32配列に変換します.

    Args:
        images (np.ndarray): 画像データのNumpy配列。各要素は(H, W, C)または(H, W)。
        shape (tuple, optional): リサイズ後の画像サイズ (H, W)。デフォルトは(640, 640)。

    Returns:
        np.ndarray: 前処理済みの画像データ (N, shape[0], shape[1], 3)。
    """
    ret = []
    for i in images:
        if i.shape[0] != shape[0] or i.shape[1] != shape[1]:
            i = cv2.resize(i, shape)
        if len(i.shape) == 2 or i.shape[-1] == 1:
            i = cv2.cvtColor(i, cv2.COLOR_GRAY2BGR)
        ret.append(i)
    ret = np.array(ret)
    if ret.shape[-1] == 1:
        ret = np.repeat(ret, 3, axis=-1)
    if len(ret.shape) == 3:
        ret = np.expand_dims(ret, axis=0)
    ret = ret.astype(np.float32) / 255.0
    return ret


def extract_features_torch(model: torch.nn.Module, images: np.ndarray, device="cpu", batch_size=16) -> np.ndarray:
    """PyTorchモデルで特徴量を抽出し、(N, D, H', W')のnumpy配列で返します.

    Args:
        model (torch.nn.Module): 特徴抽出に使用するPyTorchモデル。
        images (np.ndarray): 入力画像データ (N, H, W, C)。
        device (str, optional): "cpu"または"cuda"。デフォルトは"cpu"。
        batch_size (int, optional): バッチサイズ。デフォルトは16。

    Returns:
        np.ndarray: 抽出された特徴量マップ (N, D, H', W')。Dは特徴次元数、H', W'は特徴マップの高さと幅。
    """
    model.eval()
    features_list = []
    images_tensor = torch.from_numpy(images).float().to(device)
    if images_tensor.ndim == 3:
        images_tensor = images_tensor.unsqueeze(0)
    images_tensor = images_tensor.permute(0, 3, 1, 2)  # (N, C, H, W)
    with torch.no_grad():
        for i in range(0, images_tensor.size(0), batch_size):
            batch = images_tensor[i : i + batch_size]
            feats = model(batch)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            features_list.append(feats.cpu().numpy())
    features = np.concatenate(features_list, axis=0)
    return features  # (N, D, H, W)


def create_coreset(features: np.ndarray, budget: int = 100) -> np.ndarray:
    """K-Center Greedyアルゴリズムでコアセットを選択します.

    Args:
        features (np.ndarray): 特徴量マップ (N, D, H, W)。
        budget (int, optional): コアセットのサイズ。デフォルトは100。

    Returns:
        np.ndarray: 選択されたコアセットの特徴ベクトル (budget, D)。
    """
    N, D, H, W = features.shape
    features_flat = features.transpose(0, 2, 3, 1).reshape(N * H * W, D)  # (N*H*W, D)
    num_samples = features_flat.shape[0]
    selected_indices = [np.random.randint(0, num_samples)]
    for _ in range(budget - 1):
        dist = np.linalg.norm(features_flat - features_flat[selected_indices], axis=2)
        min_dist = np.min(dist, axis=1)
        next_index = np.argmax(min_dist)
        selected_indices.append(next_index)
        print(f"\rcreate_coreset:progress {len(selected_indices)}/{budget}", end="")
    print("\ncreate_coreset:done")
    return features_flat[selected_indices]  # (budget, D)


def save_coreset(path: str, coreset: np.ndarray):
    """コアセットを保存します."""
    np.save(path, coreset)
    print(f"save_coreset:done {path}")


def load_torch_model(model_path: str, model_class: type[torch.nn.Module], device="cpu") -> torch.nn.Module:
    """PyTorchモデルをロードします（state_dict方式）.

    Args:
        model_path (str): state_dictファイルのパス。
        model_class: モデルクラス（インスタンス化可能なもの）。
        device (str): デバイス指定 ("cpu" または "cuda")。

    Returns:
        torch.nn.Module: ロード済みモデル。

    Raises:
        FileNotFoundError: 指定されたモデルパスにファイルが存在しない場合。
        Exception: モデルのロード中にその他のエラーが発生した場合。
    """
    model = model_class()
    try:
        state_dict = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません: {model_path}")
        raise
    except Exception as e:
        print(f"エラー: モデルのロード中に問題が発生しました: {e}")
        raise
    model.load_state_dict(state_dict)
    model.eval()
    return model


def create_patchcore_model(
    model_path: str,
    model_class: type[torch.nn.Module],
    train_data_path: str,
    coreset_path: str,
    sampling: int = 500,
    budget: int = 300,
    image_shape=(640, 640),
    device="cpu",
):
    """PatchCoreモデル用コアセット生成フロー（PyTorch/aisys用）"""
    try:
        train_data = np.load(train_data_path)
    except FileNotFoundError:
        print(f"エラー: 学習データファイルが見つかりません: {train_data_path}")
        raise
    except Exception as e:
        print(f"エラー: 学習データのロード中に問題が発生しました: {e}")
        raise

    if sampling < len(train_data):
        idx = np.random.choice(len(train_data), sampling, replace=False)
        train_data = train_data[idx]
    train_data = preprocess_images(train_data, shape=image_shape)
    model = load_torch_model(model_path, model_class, device=device)
    features = extract_features_torch(model, train_data, device=device)
    coreset = create_coreset(features, budget=budget)
    save_coreset(coreset_path, coreset)
    print("PatchCoreモデル・コアセット生成が完了しました.")


def main(
    train_data_path: str,
    coreset_save_path: str,
    model_save_path: str,
    model_class: type[torch.nn.Module],
    sampling: int = 500,
    budget: int = 300,
    image_shape: tuple = (640, 640),
    device: str = "cuda",
):
    """コアセット生成のメイン関数"""
    # コアセット生成
    create_patchcore_model(
        model_path=model_save_path,
        model_class=model_class,
        train_data_path=train_data_path,
        coreset_path=coreset_save_path,
        sampling=sampling,
        budget=budget,
        image_shape=image_shape,
        device=device,
    )


def lite_converter(model_path: str, tflite_path: str, model_class: type[torch.nn.Module]):
    """PyTorchモデルをTFLite形式に変換します.

    Args:
        model_path (str): 元となるPyTorchモデルのstate_dictファイルのパス。
        tflite_path (str): 変換後のTFLiteモデルの保存パス。
        model_class (type[torch.nn.Module]): PyTorchモデルのクラス。
    """
    import torch
    import torch.utils.mobile_optimizer as mobile_optimizer

    model = load_torch_model(model_path, model_class, device="cpu")
    model.eval()
    traced_model = torch.jit.trace(model, torch.randn(1, 3, 640, 640))
    optimized_model = mobile_optimizer.optimize_for_mobile(traced_model)
    optimized_model._save_for_lite_interpreter(tflite_path)
    print(f"lite_converter:done {tflite_path}")


if __name__ == "__main__":
    # コアセット生成の実行例
    # main関数は、指定された学習データ(.npy形式)、モデル(.pt形式)を基に、
    # PatchCoreアルゴリズムで使用するコアセット(.npy形式)を生成します。
    main(
        train_data_path="./datasets/train_images.npy",  # 学習画像データがNumpy配列として保存された.npyファイルへのパス
        coreset_save_path="./datasets/patchcore_coreset.npy",  # 生成されるコアセットの保存先パス
        model_save_path="./models/mobilenet_v3_small.pt",  # 事前学習済みモデルの.ptファイルへのパス (例: torchvisionから保存したもの)
        model_class=mobilenet_v3_small,  # 使用するモデルのクラス (例: torchvision.models.mobilenet_v3_small
        sampling=500,
        budget=300,
        image_shape=(640, 640),
        device="cuda",
    )
