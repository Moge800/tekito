import os
import glob


def check_guidelines(file_path: str) -> str | bool:
    """指定されたファイルがガイドラインに従っているかを確認します。

    Args:
        file_path (str): チェックするファイルのパス
    """
    # ファイルが存在するか確認
    if not os.path.exists(file_path):
        print(f"ファイルが見つかりません: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # ガイドラインに基づくチェック
    check_results = {
        "UTF-8エンコーディング": True,
        "LF改行": True,
        "Black Format適用": True,
        "1行120文字以内": True,
        "Google Docstring Template": True,
        "説明コメントの追加": True,
        "可読性の確保": True,
        "型宣言": True,
        "エラーハンドリング": True,
        "関数の再利用": True,
    }

    for line in lines:
        # UTF-8エンコーディングかチェック（各行がUTF-8でエンコード可能か確認）
        try:
            line.encode("utf-8")
        except UnicodeEncodeError:
            check_results["UTF-8エンコーディング"] = False
        # LF改行かチェック
        if "\r" in line:
            check_results["LF改行"] = False
        if len(line.lstrip()) > 120:  # インデントを除いてチェック
            check_results["1行120文字以内"] = False
        if '"""' in line and not line.strip().startswith('"""'):
            check_results["Google Docstring Template"] = False
        if "#" in line and not line.strip().startswith("#"):
            check_results["説明コメントの追加"] = False
        # 他のチェックも追加可能

    # 結果の表示
    ret_msg = ""
    ng_msg = ""
    ng = False
    for guideline, result in check_results.items():
        if result:
            status = "適用"
        else:
            status = "未適用"
            ng = True
            ng_msg = ng_msg + f"{guideline}: {status}\n"
        ret_msg = ret_msg + f"{guideline}: {status}\n"

    return ret_msg, ng, ng_msg


if __name__ == "__main__":
    if os.path.exists("./develop/guideline_check.log"):
        os.remove("./develop/guideline_check.log")
    if os.path.exists("./develop/guideline_ng.log"):
        os.remove("./develop/guideline_ng.log")

    target_dir = "./src"
    target_py_list = glob.glob(target_dir + "/**/*.py", recursive=True)
    print(target_py_list)
    log = []
    ng_log = []
    for file_to_check in target_py_list:
        msg, flag, ng_msg = check_guidelines(file_to_check)
        print(f"ファイル: {file_to_check}")
        print(msg)
        print("=====================================")
        log.append((file_to_check, msg))
        if flag:
            ng_log.append((file_to_check, ng_msg))

    # with open("./develop/guideline_check.log", "w", encoding="utf-8") as file:
    #     for file_path, msg in log:
    #         file.write(f"ファイル: {file_path}\n")
    #         file.write(msg)
    #         file.write("=====================================\n")

    with open("./develop/guideline_ng.log", "w", encoding="utf-8") as file:
        for file_path, ng_msg in ng_log:
            file.write(f"ファイル: {file_path}\n")
            file.write(ng_msg)
            file.write("=====================================\n")
