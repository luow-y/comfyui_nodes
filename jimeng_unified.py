"""
即梦API - 统一节点（自动判断文生图/图生图）
Jimeng API - Unified Node (Auto-detect Text2Image/Image2Image)
"""

import requests
import torch
import numpy as np
from PIL import Image
import io
import tempfile
import os
import json


class JimengUnified:
    """即梦API统一节点 - 自动判断文生图或图生图，自动读取号池配置"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "a beautiful cat",
                    "multiline": True,
                    "forceInput": False,
                    "tooltip": "图像描述文本"
                }),
                "model": ([
                    "jimeng-4.0",
                    "jimeng-3.1",
                    "jimeng-3.0",
                    "jimeng-2.1",
                    "jimeng-xl-pro",
                    "nanobanana"
                ], {
                    "default": "jimeng-4.0",
                    "forceInput": False
                }),
                "ratio": ([
                    "1:1",
                    "16:9",
                    "9:16",
                    "4:3",
                    "3:4",
                    "3:2",
                    "2:3",
                    "21:9"
                ], {
                    "default": "1:1",
                    "forceInput": False
                }),
                "resolution": ([
                    "1k",
                    "2k",
                    "4k"
                ], {
                    "default": "2k",
                    "forceInput": False
                }),
                "sample_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "forceInput": False,
                    "tooltip": "采样强度"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "可选：输入图片则为图生图，不输入则为文生图"
                }),
                "image2": ("IMAGE", {
                    "tooltip": "可选：第二张输入图片"
                }),
                "image3": ("IMAGE", {
                    "tooltip": "可选：第三张输入图片"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "forceInput": False,
                    "tooltip": "随机种子，-1为随机"
                }),
                "api_url": ("STRING", {
                    "default": "http://localhost:5566",
                    "multiline": False,
                    "forceInput": False,
                    "tooltip": "API服务地址（自动添加路径）"
                }),
                "manual_session": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "forceInput": False,
                    "tooltip": "手动指定Session（为空则自动读取号池）"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    CATEGORY = "Jimeng API"
    OUTPUT_NODE = False
    
    def get_session_id(self, manual_session, api_url="http://localhost:5566"):
        """
        获取号池中的所有Session（后端会自动选择积分>5的）
        """
        if manual_session and manual_session.strip():
            print(f"🔑 使用手动指定的Session")
            return manual_session.strip()
        
        # 从后端读取号池配置
        try:
            url = f"{api_url}/admin/session-pool"
            print(f"🔗 读取号池配置...")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    sessions = data.get('data', {}).get('sessionUSList', [])
                    pool_size = len(sessions)
                    
                    if pool_size == 0:
                        raise Exception('号池为空，请先添加Session')
                    
                    # 将所有session用逗号拼接，后端会自动选择积分>=5的
                    all_sessions = ','.join(sessions)
                    print(f"✅ 读取号池成功（共{pool_size}个Session，后端将自动选择积分>=5分的）")
                    return all_sessions
                else:
                    error_msg = data.get('message', '读取号池失败')
                    print(f"❌ {error_msg}")
                    raise Exception(error_msg)
            else:
                raise Exception(f"API返回错误状态码: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ 无法连接到API服务: {api_url}")
            print(f"💡 解决方法:")
            print(f"   1. 确保即梦API服务已启动")
            print(f"   2. 检查端口是否正确: {api_url}")
            print(f"   3. 或在节点的 manual_session 参数中手动输入Session")
            raise Exception(f"无法连接到API服务: {api_url}")
        except Exception as e:
            print(f"❌ 读取号池失败: {e}")
            raise
    
    def tensor_to_pil(self, tensor):
        """将ComfyUI的tensor转换为PIL图片"""
        img_array = tensor[0].cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def pil_to_tensor(self, pil_image):
        """将PIL图片转换为ComfyUI的tensor"""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img_array)[None,]
    
    def text_to_image(self, api_url, session_id, prompt, model, ratio, 
                     resolution, sample_strength, seed):
        """文生图 - 返回所有图片URL"""
        url = f"{api_url}/v1/images/generations"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {session_id}"
        }
        
        payload = {
            "model": model,
            "prompt": prompt,
            "ratio": ratio,
            "resolution": resolution,
            "sample_strength": sample_strength
        }
        
        if seed >= 0:
            payload["seed"] = seed
        
        print(f"🌐 发送文生图请求: {url}")
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            image_urls = [item["url"] for item in result.get("data", [])]
            if image_urls:
                return image_urls  # 返回所有URLs，不只是第一个
        
        raise Exception(f"API错误 {response.status_code}: {response.text}")
    
    def image_to_image(self, api_url, session_id, prompt, images, model,
                      ratio, resolution, sample_strength):
        """图生图 - 返回所有图片URL"""
        url = f"{api_url}/v1/images/compositions"
        
        headers = {
            "Authorization": f"Bearer {session_id}"
        }
        
        # 严格的类型检查和转换
        print(f"🔍 原始 sample_strength - 类型: {type(sample_strength)}, 值: {repr(sample_strength)}")
        
        # 检查是否是函数/类型对象
        if callable(sample_strength) or isinstance(sample_strength, type):
            print(f"❌ sample_strength 是函数或类型对象，使用默认值 0.7")
            sample_strength_value = 0.7
        else:
            try:
                sample_strength_value = float(sample_strength)
            except (ValueError, TypeError) as e:
                print(f"⚠️ sample_strength 转换失败: {e}, 使用默认值 0.7")
                sample_strength_value = 0.7
        
        print(f"💪 图生图采样强度: {sample_strength_value} (类型: {type(sample_strength_value).__name__})")
        
        # 创建临时文件
        temp_files = []
        files_data = []
        
        try:
            for idx, img_tensor in enumerate(images):
                pil_img = self.tensor_to_pil(img_tensor)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                pil_img.save(temp_file.name, format='PNG')
                temp_file.close()
                temp_files.append(temp_file.name)
                
                files_data.append(('images', (f'image{idx}.png', open(temp_file.name, 'rb'), 'image/png')))
            
            data = {
                "prompt": prompt,
                "model": model,
                "ratio": ratio,
                "resolution": resolution,
                "sample_strength": str(sample_strength_value)
            }
            
            print(f"🌐 发送图生图请求: {url}")
            response = requests.post(url, headers=headers, files=files_data, data=data, timeout=300)
            
            # 关闭文件句柄
            for _, file_tuple in files_data:
                file_tuple[1].close()
            
            if response.status_code == 200:
                result = response.json()
                image_urls = [item["url"] for item in result.get("data", [])]
                if image_urls:
                    return image_urls  # 返回所有URLs，不只是第一个
            
            raise Exception(f"API错误 {response.status_code}: {response.text}")
            
        finally:
            # 清理临时文件
            for temp_file_path in temp_files:
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except Exception as e:
                    print(f"⚠️ 清理临时文件失败: {e}")
    
    def generate(self, prompt, model, ratio, resolution, sample_strength,
                image=None, image2=None, image3=None,
                seed=-1, api_url="http://localhost:5566", manual_session=""):
        """统一生成接口 - 自动判断文生图或图生图"""
        
        print(f"\n{'='*60}")
        print(f"🎨 即梦API - 统一生成节点")
        print(f"{'='*60}")
        
        try:
            # 获取Session ID
            session_id = self.get_session_id(manual_session, api_url)
            
            # 判断是文生图还是图生图
            if image is None:
                # 文生图模式
                print(f"📝 模式: 文生图 (Text to Image)")
                print(f"💬 提示词: {prompt[:50]}...")
                print(f"🎭 模型: {model}")
                print(f"📐 比例: {ratio} | 分辨率: {resolution}")
                print(f"⏳ 生成中...")
                
                image_urls = self.text_to_image(
                    api_url, session_id, prompt, model, ratio,
                    resolution, sample_strength, seed
                )
            else:
                # 图生图模式
                input_images = [image]
                if image2 is not None:
                    input_images.append(image2)
                if image3 is not None:
                    input_images.append(image3)
                
                print(f"🖼️  模式: 图生图 (Image to Image)")
                print(f"📥 输入图片数量: {len(input_images)}")
                print(f"💬 提示词: {prompt[:50]}...")
                print(f"🎭 模型: {model}")
                print(f"📐 比例: {ratio} | 分辨率: {resolution}")
                print(f"💪 采样强度: {sample_strength}")
                print(f"⏳ 生成中...")
                
                image_urls = self.image_to_image(
                    api_url, session_id, prompt, input_images, model,
                    ratio, resolution, sample_strength
                )
            
            # 下载所有生成的图片
            print(f"✅ 生成成功！共 {len(image_urls)} 张图片")
            
            output_tensors = []
            for idx, url in enumerate(image_urls):
                print(f"📥 下载第 {idx+1}/{len(image_urls)} 张图片: {url}")
                
                img_response = requests.get(url, timeout=60)
                img_response.raise_for_status()
                
                output_image = Image.open(io.BytesIO(img_response.content))
                if output_image.mode != 'RGB':
                    output_image = output_image.convert('RGB')
                
                output_tensor = self.pil_to_tensor(output_image)
                output_tensors.append(output_tensor)
                
                print(f"   尺寸: {output_image.size[0]}x{output_image.size[1]}")
            
            # 合并所有图片为batch
            batch_tensor = torch.cat(output_tensors, dim=0)
            
            # 生成信息字符串
            mode = "文生图" if image is None else f"图生图({len(input_images) if image is not None else 0}张)"
            all_urls = "\n".join(image_urls)
            info = f"模式: {mode} | 模型: {model} | {ratio} {resolution}\n生成数量: {len(image_urls)}张\nURLs:\n{all_urls}"
            
            print(f"✨ 完成！共下载 {len(image_urls)} 张图片")
            print(f"{'='*60}\n")
            
            return (batch_tensor, info)
            
        except requests.exceptions.Timeout:
            error_msg = "请求超时，请检查网络或API服务"
            print(f"❌ {error_msg}")
            print(f"{'='*60}\n")
            blank = torch.zeros((1, 512, 512, 3))
            return (blank, f"ERROR: {error_msg}")
            
        except requests.exceptions.ConnectionError:
            error_msg = "连接失败，请确认API服务已启动"
            print(f"❌ {error_msg}")
            print(f"{'='*60}\n")
            blank = torch.zeros((1, 512, 512, 3))
            return (blank, f"ERROR: {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ 生成失败: {error_msg}")
            print(f"{'='*60}\n")
            blank = torch.zeros((1, 512, 512, 3))
            return (blank, f"ERROR: {error_msg}")


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "Jimeng_Unified": JimengUnified
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Jimeng_Unified": "Jimeng (即梦-智能生图)"
}

# 用于直接导入
if __name__ == "__main__":
    print("此文件是ComfyUI节点，请在ComfyUI中使用")

