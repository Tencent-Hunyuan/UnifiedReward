#!/bin/bash
# 一个稳健、高效的数据集自动化准备脚本。
# 经过优化，解决了原脚本的稳定性与效率问题，并能整合处理多个数据源。

# --- 脚本安全设置 ---
# 任何命令返回非零值（表示错误），脚本将立即退出，防止错误继续。
set -e
# 如果脚本中使用了未定义的变量，将报错并退出。
set -u

# --- 配置区 ---
readonly URL_LIST="urls.txt"                # 包含分片下载链接的文件
readonly OTHER_ZIPS=("images_HPD.zip" "images_OIP.zip") # 其他需要直接解压的zip包
readonly DOWNLOAD_DIR="temp_downloads"      # 用于存放下载过程中的所有临时文件
readonly FINAL_IMAGE_DIR="images"           # 最终存放所有图片的目录名

# --- 主逻辑开始 ---
echo "--- 数据集自动化准备脚本 (全整合版) ---"
echo "开始前，清理旧的残留文件，确保环境干净..."

# 安全地删除旧目录，并创建新的目录
rm -rf "$DOWNLOAD_DIR"
rm -rf "$FINAL_IMAGE_DIR"
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$FINAL_IMAGE_DIR" # <-- 直接在此创建最终目录

# --- 阶段一：下载文件分片 (如果urls.txt存在) ---
echo
if [ -f "$URL_LIST" ]; then
    echo "=== 阶段一：从 $URL_LIST 下载所有文件分片... ==="
    echo "将尝试使用 'aria2c' 进行高速下载。如果未安装，则自动降级为 'wget'。"

    while IFS= read -r url || [ -n "$url" ]; do
        filename=$(basename "${url%%\?*}")
        echo "正在下载: ${filename}"
        if command -v aria2c &> /dev/null; then
            aria2c --console-log-level=warn -c -x 16 -s 16 -k 1M -d "$DOWNLOAD_DIR" -o "$filename" "$url"
        else
            wget -q -c -O "$DOWNLOAD_DIR/$filename" "$url"
        fi
    done < "$URL_LIST"
    echo "所有分片文件下载成功。"
else
    echo "--- 跳过阶段一：未找到 $URL_LIST 文件。 ---"
fi


# --- 阶段二：合并文件分片 (如果分片存在) ---
echo
if ls "$DOWNLOAD_DIR"/images.zip.part-* 1> /dev/null 2>& 1; then
    echo "=== 阶段二：合并所有分片文件... ==="
    echo "正在将所有 part-* 文件合并成一个完整的 images.zip..."
    cat "$DOWNLOAD_DIR"/images.zip.part-* > "$DOWNLOAD_DIR"/images.zip
    echo "images.zip 合并成功。"
else
    echo "--- 跳过阶段二：未找到分片文件。 ---"
fi

# --- 阶段三：解压所有压缩包 ---
echo
echo "=== 阶段三：解压所有压缩包到 '${FINAL_IMAGE_DIR}' 文件夹... ==="

# 检查unzip命令是否存在
if ! command -v unzip &> /dev/null; then
    echo "错误: 未找到 'unzip' 命令。请先安装它 (例如: 'sudo apt-get install unzip')。脚本中止。"
    exit 1
fi

# 定义一个解压函数，避免代码重复
unzip_if_exists() {
    local zip_file_path=$1
    if [ -f "$zip_file_path" ]; then
        echo "正在处理: $zip_file_path"
        if ! unzip -tq "$zip_file_path"; then
            echo "错误: $zip_file_path 文件似乎已损坏或无效。跳过此文件。"
            return
        fi
        # 直接解压到最终目录。-n: 不覆盖已存在的文件，更安全
        unzip -nqo "$zip_file_path" -d "$FINAL_IMAGE_DIR"
        echo "$zip_file_path 解压完成。"
    else
        echo "信息: 未找到 $zip_file_path，跳过。"
    fi
}

# 1. 处理由分片合并而来的主zip包
unzip_if_exists "$DOWNLOAD_DIR/images.zip"

# 2. 循环处理在当前目录下的其他zip包
for zip_file in "${OTHER_ZIPS[@]}"; do
    unzip_if_exists "$zip_file"
done

echo "所有压缩包处理完毕。"

# --- 阶段四：最终清理 ---
echo
echo "=== 阶段四：清理临时文件... ==="
rm -rf "$DOWNLOAD_DIR"
echo "临时文件清理完毕。"

echo -e "\n--- ✅ 全部完成！所有图片数据已整合到 '${FINAL_IMAGE_DIR}' 文件夹中。 ---"