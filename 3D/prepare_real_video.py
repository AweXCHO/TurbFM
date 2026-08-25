import os
import cv2
import tempfile
import shutil
import time

def resize_video(input_path, output_path, target_size=(224, 224)):
    """
    调整视频大小并保存
    :param input_path: 输入视频路径
    :param output_path: 输出视频路径
    :param target_size: 目标尺寸 (width, height)
    """
    start_time = time.time()
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"\n❌ 无法打开视频文件: {input_path}")
        return False
    
    # 获取原视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n" + "="*80)
    print(f"🎬 正在处理视频: {input_path}")
    print(f"📐 原尺寸: {width}x{height} -> 目标尺寸: {target_size[0]}x{target_size[1]}")
    print(f"📊 视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 时长={duration:.2f}秒")
    
    # 获取视频编解码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
    
    # 创建视频写入器
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    if not out.isOpened():
        print(f"无法创建输出视频: {output_path}")
        cap.release()
        return False
    
    # 逐帧处理
    processed_frames = 0
    progress_interval = max(1, total_frames // 10)  # 每10%输出一次进度
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 调整帧大小
        resized_frame = cv2.resize(frame, target_size)
        
        # 写入帧
        out.write(resized_frame)
        processed_frames += 1
        
        # 显示进度（每处理10%或每100帧输出一次）
        if processed_frames % progress_interval == 0 or processed_frames % 100 == 0:
            progress = (processed_frames / total_frames) * 100
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / processed_frames) * (total_frames - processed_frames) if processed_frames > 0 else 0
            print(f"⏳ 进度: [{progress:6.2f}%] {processed_frames}/{total_frames}帧 | 已用时: {elapsed_time:.2f}s | 预计剩余: {eta:.2f}s")
    
    # 释放资源
    cap.release()
    out.release()
    
    elapsed_time = time.time() - start_time
    print(f"✅ 处理完成！新尺寸: {target_size[0]}x{target_size[1]} | 总用时: {elapsed_time:.2f}秒")
    print("="*80)
    return True

def process_videos_in_directory(base_dir, target_size=(224, 224)):
    """
    处理指定目录下所有子文件夹中的视频文件
    :param base_dir: 基础目录
    :param target_size: 目标尺寸
    """
    video_extensions = ('.mp4', '.avi')
    
    # 首先收集所有视频文件
    video_files = []
    skipped_dirs = []
    scanned_dirs = []
    
    print(f"\n🔍 开始扫描目录: {base_dir}")
    
    for root, dirs, files in os.walk(base_dir):
        # 跳过__pycache__、experiment、model、para、utils目录
        if '__pycache__' in root or 'experiment' in root or 'model' in root or 'para' in root or 'utils' in root:
            skipped_dirs.append(root)
            continue
        
        scanned_dirs.append(root)
        print(f"   扫描目录: {root}")
        
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                video_files.append(video_path)
                print(f"      ✅ 找到视频: {file}")
    
    total_videos = len(video_files)
    print(f"\n📁 扫描完成！")
    print(f"   扫描目录数: {len(scanned_dirs)}")
    print(f"   跳过目录数: {len(skipped_dirs)}")
    print(f"   发现视频数: {total_videos}")
    
    if skipped_dirs:
        print(f"   跳过的目录: {skipped_dirs}")
    
    print("-"*80)
    
    if total_videos == 0:
        print("⚠️ 警告：没有找到任何视频文件！")
        print(f"   检查目录: {base_dir}")
        print(f"   支持的扩展名: {video_extensions}")
        return
    
    # 处理所有视频
    success_count = 0
    fail_count = 0
    
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n📝 [{idx}/{total_videos}] 准备处理: {video_path}")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # 处理视频到临时文件
            if resize_video(video_path, temp_path, target_size):
                # 替换原文件
                os.remove(video_path)
                shutil.move(temp_path, video_path)
                print(f"🔄 已覆盖原文件: {video_path}")
                success_count += 1
            else:
                # 删除临时文件
                os.remove(temp_path)
                fail_count += 1
        except Exception as e:
            print(f"❌ 处理视频时出错 {video_path}: {e}")
            fail_count += 1
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # 输出汇总统计
    print("\n" + "="*80)
    print(f"📊 处理汇总:")
    print(f"   总数: {total_videos} | 成功: {success_count} | 失败: {fail_count}")
    print("="*80)

if __name__ == "__main__":
    # 基础目录 - 处理83文件夹下的所有子文件夹
    base_directory = "/ayt1/anyitong/TurbMAE/3D/83"
    
    print("\n" + "="*80)
    print("🎥 视频批量处理工具")
    print("="*80)
    print(f"📂 处理目录: {base_directory}")
    print(f"🎯 目标尺寸: {224}x{224}")
    print(f"🕐 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    start_total = time.time()
    process_videos_in_directory(base_directory, target_size=(224, 224))
    
    total_time = time.time() - start_total
    print(f"\n🎉 全部处理完成！总耗时: {total_time:.2f}秒")
    print("="*80)