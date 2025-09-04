import cv2
import numpy as np

image = cv2.imread("./testpy/2024-02-16-044645_1280x1024_scrot.png")

# 画像を1280x1024にリサイズ (元画像のサイズが異なる場合を想定)
# このリサイズでも、元画像の解像度によっては情報が変化します。
original_image = cv2.resize(image, (1280, 1024), interpolation=cv2.INTER_AREA)

# ESPCN x4を適用 (仮想的な超解像処理として、ここではCubic補間による4倍拡大を模擬)
# cv2.INTER_CUBICによる拡大は、失われた詳細を完全に復元するものではありません。
# あくまで周囲のピクセルから補間するため、元の鮮明さには及ばないことがあります。
magnification = 4
super_resolved = cv2.resize(
    original_image,
    (original_image.shape[1] * magnification, original_image.shape[0] * magnification),  # (1280*4, 1024*4) に拡大
    interpolation=cv2.INTER_CUBIC,
)

# 元の目標サイズに戻す (1280x1024)
# 再度縮小するため、ここでも情報の間引きが発生し、画質に影響します。
# cv2.INTER_LANCZOS4 は、cv2.INTER_AREAと比較してシャープな結果が得られることがあります。
final_image = cv2.resize(super_resolved, (1280, 1024), interpolation=cv2.INTER_LANCZOS4)

# # 保存
# cv2.imwrite("output_image.jpg", final_image)
# print("処理が完了しました。output_image.jpg に保存されました。")

show_image = cv2.hconcat([original_image, final_image])
cv2.imshow("Image Processing", show_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
