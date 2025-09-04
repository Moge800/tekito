from PIL import Image, ImageFile
import numpy as np
import cv2
import io
from pathlib import Path
from typing import List, Optional

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 破損JPEGにも対応


def repair_and_save_jpeg(input_path: Path, output_dir: Path) -> Optional[Path]:
    """破損したJPEG画像を修復し保存する

    Args:
        input_path: 入力画像のパス
        output_dir: 出力先ディレクトリのパス

    Returns:
        出力ファイルのパス、失敗した場合はNone
    """
    try:
        # 画像をバイナリとして読み込み
        with open(input_path, "rb") as f:
            buff = io.BytesIO(f.read())
            img_pil = Image.open(buff)
            img_np = np.array(img_pil, dtype=np.uint8)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 出力ファイルパスを作成して保存
        output_path = output_dir / input_path.name
        cv2.imwrite(str(output_path), img_cv)
        print(f"処理完了: {output_path}")
        return output_path

    except Exception as e:
        print(f"画像処理エラー ({input_path}): {e}")
        return None


def process_directory(target_dir: str, output_dir: str) -> List[Path]:
    """指定ディレクトリ内のJPEG画像をすべて処理する

    Args:
        target_dir: 対象ディレクトリのパス
        output_dir: 出力先ディレクトリのパス

    Returns:
        処理成功した画像パスのリスト
    """
    # 出力先ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 対象ファイルの検索
    target_files = list(Path(target_dir).glob("*.jpg"))
    print(f"処理対象: {len(target_files)}ファイル")

    # 各ファイルの処理
    successful_paths = []
    for file_path in target_files:
        result = repair_and_save_jpeg(file_path, output_path)
        if result:
            successful_paths.append(result)

    print(f"JPEG修復・保存完了。成功: {len(successful_paths)}/{len(target_files)}。出力先: {output_path}")
    return successful_paths


if __name__ == "__main__":
    process_directory("./target_dir", "./output_dir")
